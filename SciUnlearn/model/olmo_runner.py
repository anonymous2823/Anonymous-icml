import re
from typing import Dict, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import AppConfig


ALLOWED_AR = [
    "A is True, R is True, and R explains A",
    "A is True, R is True, but R does not explain A",
    "A is True, R is False",
    "A is False, R is True",
    "A is False, R is False",
]


class OLMORunner:
    """
    Loads OLMo once and provides helper methods to answer generated QA items.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        print(f"[INFO] Loading OLMo model: {config.olmo_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.olmo_model_name)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.olmo_model_name,
            device_map=config.olmo_device_map,
            torch_dtype=dtype_map.get(config.olmo_dtype, torch.float16),
            trust_remote_code=True,
        )
        self.model.eval()
        print("[INFO] OLMo model loaded successfully.")

    # ---------------- Prompt builders ----------------

    @staticmethod
    def build_messages_mcq(question: str) -> List[Dict[str, str]]:
        system = (
            "You are a precise scientific assistant. Read the question and its embedded options "
            "[A], [B], [C], [D]. Respond with only the single correct letter A, B, C, or D. "
            "Do not include any other text."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

    @staticmethod
    def build_messages_tf(question: str) -> List[Dict[str, str]]:
        system = (
            "You are a precise scientific assistant. Respond with only one word: True or False. "
            "Do not include any other text."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

    @staticmethod
    def build_messages_fill(question: str) -> List[Dict[str, str]]:
        system = (
            "You are a precise scientific assistant. The question contains exactly one blank represented "
            "by ______. Immediately after the blank, two options are provided in parentheses in the format: "
            "_______(option1/option2). You MUST choose and respond with ONLY ONE of the two given options "
            "inside the parentheses. Do not rephrase, do not add explanation. "
            "Respond with exactly the option text as it appears, nothing more."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

    @staticmethod
    def build_messages_ar(question: str) -> List[Dict[str, str]]:
        system = (
            "You are a precise scientific assistant. Choose exactly one label from the set: "
            "'A is True, R is True, and R explains A' | "
            "'A is True, R is True, but R does not explain A' | "
            "'A is True, R is False' | 'A is False, R is True' | 'A is False, R is False'. "
            "Respond with the chosen label verbatim and nothing else."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

    # ---------------- Post-processing ----------------

    @staticmethod
    def post_mcq(text: str) -> str:
        m = re.search(r"\b([ABCD])\b", text.strip(), flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        t = text.strip().lower()
        for k, v in {"a": "A", "b": "B", "c": "C", "d": "D"}.items():
            if t.startswith(k):
                return v

        return (text.strip()[:1] or "A").upper()

    @staticmethod
    def post_tf(text: str) -> str:
        t = text.strip().lower()
        if "true" in t and "false" not in t:
            return "True"
        if "false" in t and "true" not in t:
            return "False"

        tok = re.split(r"\s+", text.strip())[0] if text.strip() else ""
        return "True" if tok.lower().startswith("t") else "False"

    @staticmethod
    def post_fill(text: str) -> str:
        ans = text.strip().strip('"\'').strip()
        ans = re.sub(r"\s+", " ", ans)
        return ans[:200]

    @staticmethod
    def post_ar(text: str) -> str:
        t = text.strip()

        for label in ALLOWED_AR:
            if t == label:
                return t

        tl = t.lower()
        for label in ALLOWED_AR:
            if label.lower() in tl:
                return label

        return ALLOWED_AR[0]

    # ---------------- Generation ----------------

    def generate(self, messages: List[Dict[str, str]]) -> str:
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.olmo_max_new_tokens,
                do_sample=self.config.olmo_do_sample,
                temperature=self.config.olmo_temperature if self.config.olmo_do_sample else None,
                top_p=self.config.olmo_top_p if self.config.olmo_do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        text = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return text.strip()

    # ---------------- QA enrichment ----------------

    def answer_item(self, qa_type: str, question: str) -> str:
        """
        Generate OLMo answer for one QA item and normalize it.
        """
        if qa_type == "mcq":
            raw = self.generate(self.build_messages_mcq(question))
            return self.post_mcq(raw)

        if qa_type == "true_false":
            raw = self.generate(self.build_messages_tf(question))
            return self.post_tf(raw)

        if qa_type == "fill_blank":
            raw = self.generate(self.build_messages_fill(question))
            return self.post_fill(raw)

        if qa_type == "assertion_reason":
            raw = self.generate(self.build_messages_ar(question))
            return self.post_ar(raw)

        raise ValueError(f"Unsupported qa_type: {qa_type}")

    def answer_item_with_consistency_check(self, qa_type: str, question: str, num_runs: int = 3) -> Dict[str, Any]:
        """
        Generate OLMo answer for one QA item multiple times and check consistency.
        
        Returns:
            Dict with keys:
            - 'answer': The final answer (same across all runs if consistent)
            - 'all_answers': List of all answers from each run
            - 'is_consistent': True if all answers are identical, False otherwise
            - 'consistency_score': Fraction of runs that matched the final answer
        """
        answers = []
        for _ in range(num_runs):
            answer = self.answer_item(qa_type, question)
            answers.append(answer)
        
        # Check if all answers are identical
        is_consistent = len(set(answers)) == 1
        
        # The answer is the first one (or any since they're all the same if consistent)
        final_answer = answers[0] if answers else ""
        
        # Calculate consistency score: how many answers match the final answer
        matching_count = sum(1 for a in answers if a == final_answer)
        consistency_score = matching_count / num_runs if num_runs > 0 else 0.0
        
        return {
            "answer": final_answer,
            "all_answers": answers,
            "is_consistent": is_consistent,
            "consistency_score": consistency_score,
        }

    def enrich_qa_by_claim(self, qa_by_claim: List[Dict[str, Any]], use_consistency_check: bool = False, num_runs: int = 3) -> List[Dict[str, Any]]:
        """
        Add `olmo_answer` to every QA item in every claim object.
        
        Args:
            qa_by_claim: List of claim objects with QA items
            use_consistency_check: If True, validate answer consistency across multiple runs
            num_runs: Number of times to run each question (only used if use_consistency_check=True)
        
        Mutates and returns the same structure for convenience.
        """
        for claim_obj in qa_by_claim:
            for item in claim_obj.get("mcq", []) or []:
                if use_consistency_check:
                    self._enrich_item_with_consistency(item, "mcq", num_runs)
                else:
                    item["olmo_answer"] = self.answer_item("mcq", item.get("question", ""))

            for item in claim_obj.get("true_false", []) or []:
                if use_consistency_check:
                    self._enrich_item_with_consistency(item, "true_false", num_runs)
                else:
                    item["olmo_answer"] = self.answer_item("true_false", item.get("question", ""))

            for item in claim_obj.get("fill_blank", []) or []:
                if use_consistency_check:
                    self._enrich_item_with_consistency(item, "fill_blank", num_runs)
                else:
                    item["olmo_answer"] = self.answer_item("fill_blank", item.get("question", ""))

            for item in claim_obj.get("assertion_reason", []) or []:
                if use_consistency_check:
                    self._enrich_item_with_consistency(item, "assertion_reason", num_runs)
                else:
                    item["olmo_answer"] = self.answer_item("assertion_reason", item.get("question", ""))

        return qa_by_claim

    def _enrich_item_with_consistency(self, item: Dict[str, Any], qa_type: str, num_runs: int = 3) -> None:
        """
        Helper to enrich a single QA item with consistency-checked answer.
        Validates that all runs produce the same answer (consistency check).
        
        Modifies item in place.
        """
        question = item.get("question", "")
        result = self.answer_item_with_consistency_check(qa_type, question, num_runs)
        
        # Store the main answer
        item["olmo_answer"] = result["answer"]
        
        # Store consistency metadata for later filtering
        item["olmo_all_answers"] = result["all_answers"]
        item["olmo_is_consistent"] = result["is_consistent"]
        item["olmo_consistency_score"] = result["consistency_score"]