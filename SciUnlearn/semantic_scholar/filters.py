from typing import Any, Dict
from config import AppConfig
from llm_client.azure_gpt5_client import (
    call_llm_is_computer_science_paper,
    call_llm_paper_type
)


# def is_computer_science_paper(meta: Dict[str, Any]) -> bool:
#     """
#     Return True if the paper belongs to the Computer Science domain.
#     Checks both `fieldsOfStudy` and `s2FieldsOfStudy` to be robust against API variations.
#     """
#     fos = meta.get("fieldsOfStudy") or []
#     for item in fos:
#         if isinstance(item, str) and item.strip().lower() == "computer science":
#             return True
#         if isinstance(item, dict):
#             val = (
#                 item.get("category")
#                 or item.get("name")
#                 or item.get("field")
#                 or item.get("displayName")
#             )
#             if isinstance(val, str) and val.strip().lower() == "computer science":
#                 return True

#     s2fos = meta.get("s2FieldsOfStudy") or []
#     for item in s2fos:
#         if isinstance(item, str) and item.strip().lower() == "computer science":
#             return True
#         if isinstance(item, dict):
#             val = (
#                 item.get("category")
#                 or item.get("name")
#                 or item.get("field")
#                 or item.get("displayName")
#             )
#             if isinstance(val, str) and val.strip().lower() == "computer science":
#                 return True

#     return False



# def is_computer_science_paper(meta: Dict[str, Any], config: AppConfig) -> bool:
#     """
#     Classify whether a paper is Computer Science using GPT on:
#     - title + abstract, if abstract exists
#     - title only, if abstract is missing
#     """
#     title = (meta.get("title") or "").strip()
#     abstract = (meta.get("abstract") or "").strip()

#     if not title:
#         print("[FILTER][CS] Rejecting paper because title is missing.")
#         return False

#     if not abstract:
#         if getattr(config, "gpt_filter_title_only_fallback", True):
#             print(f"[FILTER][CS] Abstract missing for '{title[:120]}'; using title-only classification.")
#         else:
#             print(f"[FILTER][CS] Rejecting '{title[:120]}' because abstract is missing.")
#             return False

#     if not config.enable_gpt_cs_filter:
#         return True

#     try:
#         result = call_llm_is_computer_science_paper(
#             title=title,
#             abstract=abstract,   # may be empty; GPT function will still receive title
#             config=config,
#         )

#         is_cs = bool(result.get("is_computer_science", False))
#         confidence = result.get("confidence", "unknown")
#         reason = result.get("reason", "")

#         print(
#             f"[FILTER][CS] title='{title[:120]}' | "
#             f"is_cs={is_cs} | confidence={confidence} | reason={reason}"
#         )

#         return is_cs

#     except Exception as e:
#         print(f"[FILTER][CS] GPT classification failed for '{title[:120]}': {e}")
#         return False



def metadata_says_computer_science(meta: Dict[str, Any]) -> bool:
    """
    Step 1 of CS filtering:
    Check whether Semantic Scholar metadata says the paper belongs to Computer Science.

    Supports common shapes such as:
    - fieldsOfStudy = ["Computer Science", ...]
    - s2FieldsOfStudy = [{"category": "Computer Science"}, ...]
    """
    # Case 1: plain list of strings
    fos = meta.get("fieldsOfStudy") or []
    print("Metadata fieldsOfStudy:", fos)
    for x in fos:
        if isinstance(x, str) and x.strip().lower() == "computer science":
            return True

    # Case 2: richer list of dicts
    s2_fos = meta.get("s2FieldsOfStudy") or []
    print("Metadata s2FieldsOfStudy:", s2_fos)
    for x in s2_fos:
        if isinstance(x, dict):
            cat = (x.get("category") or "").strip().lower()
            if cat == "computer science":
                return True

    return False


def is_computer_science_paper(meta: Dict[str, Any], config: AppConfig) -> bool:
    """
    Two-step CS filter:

    Step 1:
      Require Semantic Scholar metadata to indicate Computer Science.

    Step 2:
      If metadata passes, verify again using GPT on:
      - title + abstract if abstract exists
      - title only if abstract is missing
    """
    title = (meta.get("title") or "").strip()
    abstract = (meta.get("abstract") or "").strip()

    if not title:
        print("[FILTER][CS] Rejecting paper because title is missing.")
        return False

    # -------------------------
    # Step 1: metadata gate
    # -------------------------
    if not metadata_says_computer_science(meta):
        print(f"[FILTER][CS] Rejecting '{title[:120]}' because metadata does not indicate Computer Science.")
        return False

    print(f"[FILTER][CS] Metadata says Computer Science for '{title[:120]}'; proceeding to GPT verification.")

    # -------------------------
    # Step 2: GPT verification
    # -------------------------
    if not abstract:
        print(f"[FILTER][CS] Abstract missing for '{title[:120]}'; using title-only GPT classification.")

    if not config.enable_gpt_cs_filter:
        return True

    try:
        result = call_llm_is_computer_science_paper(
            title=title,
            abstract=abstract,   # may be empty; GPT will fall back to title
            config=config,
        )

        is_cs = bool(result.get("is_computer_science", False))
        confidence = result.get("confidence", "unknown")
        reason = result.get("reason", "")

        print(
            f"[FILTER][CS] GPT verification | title='{title[:120]}' | "
            f"is_cs={is_cs} | confidence={confidence} | reason={reason}"
        )

        return is_cs

    except Exception as e:
        print(f"[FILTER][CS] GPT verification failed for '{title[:120]}': {e}")
        return False



def get_paper_type(meta: Dict[str, Any], config: AppConfig) -> str:
    """
    Predict one paper type from:
    - title + abstract, if abstract exists
    - title only, if abstract is missing
    """
    title = (meta.get("title") or "").strip()
    abstract = (meta.get("abstract") or "").strip()

    if not title:
        print("[FILTER][TYPE] Rejecting because title is missing.")
        return "other"

    if not abstract:
        if getattr(config, "gpt_filter_title_only_fallback", True):
            print(f"[FILTER][TYPE] Abstract missing for '{title[:120]}'; using title-only paper-type classification.")
        else:
            print(f"[FILTER][TYPE] Rejecting '{title[:120]}' because abstract is missing.")
            return "other"

    if not config.enable_gpt_paper_type_filter:
        return config.required_paper_type

    try:
        result = call_llm_paper_type(
            title=title,
            abstract=abstract,   # may be empty; GPT function will still receive title
            config=config,
        )

        paper_type = (result.get("paper_type") or "other").strip()
        confidence = result.get("confidence", "unknown")
        reason = result.get("reason", "")

        print(
            f"[FILTER][TYPE] title='{title[:120]}' | "
            f"paper_type={paper_type} | confidence={confidence} | reason={reason}"
        )

        return paper_type

    except Exception as e:
        print(f"[FILTER][TYPE] GPT paper-type classification failed for '{title[:120]}': {e}")
        return "other"



def is_experimental_original_research_paper(meta: Dict[str, Any], config: AppConfig) -> bool:
    """
    Keep only papers classified as experimental_original_research.
    """
    predicted_type = get_paper_type(meta, config)
    return predicted_type == config.required_paper_type