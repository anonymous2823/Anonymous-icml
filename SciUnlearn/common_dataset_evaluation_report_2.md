# Common-Dataset Evaluation Report

## 1. Input Files
- **forget**: `forget_common_3.json`
- **retain_external**: `retain_external_common_3.json`
- **retain_internal**: `retain_internal_common_3.json`
- **derived**: `derived_common_3.json`
- **forget_rejected**: `forget_olmo_rejected_3.json`
- **retain_external_rejected**: `retain_external_olmo_rejected_3.json`
- **retain_internal_rejected**: `retain_internal_olmo_rejected_3.json`
- **derived_rejected**: `derived_olmo_rejected_3.json`
- **year_index_dir**: `year_wise_corpus_ids`

## 2. Dataset Summaries
### forget
- Record count: **76**
- Total claims: **170**
- Question counts — MCQ: **276**, TF: **202**, Fill: **254**, AR: **340**, Total: **1072**
- Rejected QA counts — MCQ: **47**, TF: **85**, Fill: **59**, AR: **0**, Total: **191**
- Rejected questions per paper — Mean: **3.6037735849056602**, Median: **3.0**, Std: **2.830691779183351**, Min: **1.0**, Max: **12.0**
- Year coverage: `{'2022': 13, '2021': 18, '2020': 16, '2019': 15, '2018': 14}`

### retain_external
- Record count: **76**
- Total claims: **171**
- Question counts — MCQ: **284**, TF: **180**, Fill: **254**, AR: **342**, Total: **1060**
- Rejected QA counts — MCQ: **42**, TF: **98**, Fill: **59**, AR: **0**, Total: **199**
- Rejected questions per paper — Mean: **3.3728813559322033**, Median: **3.0**, Std: **2.3208582945134157**, Min: **1.0**, Max: **10.0**
- Retain paper year coverage (Semantic Scholar): `{'2022': 5, '2013': 2, '2020': 6, '2021': 3, '2019': 7, '2015': 9, '2016': 8, '2018': 8, '2011': 4, '2017': 9, '2006': 2, '2012': 2, '2014': 4, '2001': 1, '2010': 1, '2008': 1, '2009': 2, '2004': 1, '1998': 1}`

### retain_internal
- Record count: **76**
- Total claims: **170**
- Question counts — MCQ: **129**, TF: **126**, Fill: **11**, AR: **152**, Total: **418**
- Rejected QA counts — MCQ: **23**, TF: **26**, Fill: **141**, AR: **0**, Total: **190**
- Rejected questions per paper — Mean: **2.533333333333333**, Median: **2.0**, Std: **0.9140872800534725**, Min: **1.0**, Max: **5.0**
- Year coverage: `{'2022': 13, '2021': 18, '2020': 16, '2019': 15, '2018': 14}`

### derived
- Record count: **76**
- Total claims: **170**
- Question counts — MCQ: **173**, TF: **173**, Fill: **12**, AR: **228**, Total: **586**
- Rejected QA counts — MCQ: **55**, TF: **55**, Fill: **216**, AR: **0**, Total: **326**
- Rejected questions per paper — Mean: **4.2894736842105265**, Median: **4.0**, Std: **1.326012914457276**, Min: **2.0**, Max: **8.0**
- Year coverage: `{'2022': 13, '2021': 18, '2020': 16, '2019': 15, '2018': 14}`

## 3. Forget vs External Retain Paper Overlap
- Forget paper count: **76**
- External retain paper count: **76**
- Overlap count: **0**
- Has overlap: **False**
- Overlap ID examples: `[]`

## 4. Cost Statistics
### Overall
- Event count: **2634**
- Total cost sum: **41.25308624999993** | mean: **0.01566176395216401** | median: **0.0036637500000000003** | std: **0.01893465560199232**
- Prompt cost sum: **6.488706249999979** | mean: **0.002463442008352316**
- Completion cost sum: **34.76438000000008** | mean: **0.013198321943811693**

### forget
- Event count: **1086**
- Total cost sum: **27.55022750000001** | mean: **0.025368533609576426** | median: **0.0276175** | std: **0.02146769341330877**
- Prompt cost sum: **4.701217499999999** | mean: **0.00432892955801105**
- Completion cost sum: **22.849010000000018** | mean: **0.021039604051565377**
- By type:
  - `claim_extraction` -> count: **102**, sum: **2.729378749999999**, mean: **0.02675861519607843**, median: **0.027046874999999998**
  - `cs_paper_filter` -> count: **262**, sum: **0.9269812500000001**, mean: **0.003538096374045802**, median: **0.003054375**
  - `derived_question_generation` -> count: **100**, sum: **4.485651249999998**, mean: **0.0448565125**, median: **0.043436875**
  - `paper_type_filter` -> count: **202**, sum: **0.6377537500000002**, mean: **0.0031571967821782177**, median: **0.00279125**
  - `qa_generation` -> count: **217**, sum: **8.719622500000003**, mean: **0.04018259216589862**, median: **0.04005625**
  - `retain_internal_generation` -> count: **100**, sum: **4.11165125**, mean: **0.0411165125**, median: **0.041184374999999995**
  - `verbatim_claim_extraction` -> count: **103**, sum: **5.939188750000003**, mean: **0.057662026699029124**, median: **0.057055**

### retain_external
- Event count: **1548**
- Total cost sum: **13.702858750000004** | mean: **0.008851975936692506** | median: **0.0029950000000000003** | std: **0.013200370606379927**
- Prompt cost sum: **1.7874887499999998** | mean: **0.0011547084948320414**
- Completion cost sum: **11.915369999999967** | mean: **0.007697267441860466**
- By type:
  - `claim_extraction` -> count: **75**, sum: **2.01701**, mean: **0.026893466666666668**, median: **0.02727375**
  - `cs_paper_filter` -> count: **665**, sum: **2.0302699999999985**, mean: **0.0030530375939849625**, median: **0.00276625**
  - `paper_type_filter` -> count: **607**, sum: **1.8222124999999996**, mean: **0.003001997528830313**, median: **0.002925**
  - `qa_generation` -> count: **201**, sum: **7.83336625**, mean: **0.03897197139303483**, median: **0.037063750000000006**

### retain_internal
- Event count: **0**
- Total cost sum: **0.0** | mean: **None** | median: **None** | std: **None**
- Prompt cost sum: **0.0** | mean: **None**
- Completion cost sum: **0.0** | mean: **None**

### derived
- Event count: **0**
- Total cost sum: **0.0** | mean: **None** | median: **None** | std: **None**
- Prompt cost sum: **0.0** | mean: **None**
- Completion cost sum: **0.0** | mean: **None**

## 5. Forget Q1/Q2 Balance
- Total QA groups inspected: **536**
- Q1 counts: `{'mcq': 138, 'true_false': 101, 'fill_blank': 127, 'assertion_reason': 170, 'total': 536}`
- Q2 counts: `{'mcq': 138, 'true_false': 101, 'fill_blank': 127, 'assertion_reason': 170, 'total': 536}`
- Odd QA groups count: **0**