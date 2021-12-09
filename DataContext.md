# Explanatory variables:

Categorical prefeaced with *C*, 
Numerical prefaced with *N* 
(! denotes items which should be looked into further): 

- emotional
    - C Irritability | Instance 0 
    - C Miserableness | Instance 0
    - C Mood swings | Instance 0
    - C Sensitivity / hurt feelings | Instance 0
    - C Worrier / anxious feelings | Instance 0
- dietary 
    - C! Cooked vegetable intake | Instance 0
    - C! Fresh fruit intake | Instance 0
    - C! Oily fish intake | Instance 0
    - C Salad / raw vegetable intake | Instance 0
    - C Salt added to food | Instance 0
    - C! Tea intake | Instance 0
    - C! Water intake | Instance 0
- socio
    - C! Average total household income before tax | Instance 0
    - C Ethnic background | Instance 0
    - C! Number of vehicles in household | Instance 0
    - C Qualifications | Instance 0
    - C Sex
    - C Type of accommodation lived in | Instance 0
    - N Age at recruitment
    - N Crime score
    - N Education score
    - N Employment score
    - N Housing score
    - N Income score
    - N Index of Multiple Deprivation
    - N! Living environment
    - N Townsend deprivation index at recruitment
- physical
    - C Above moderate/vigorous recommendation | Instance 0
    - C Above moderate/vigorous/walking recommendation | Instance 0
    - C Attendance/disability/mobility allowance | Instance 0
    - C! Duration of walks | Instance 0
    - C! Frequency of stair climbing in last 4 weeks | Instance 0
    - C! Frequency of tiredness / lethargy in last 2 weeks | Instance 0
    - C IPAQ activity group | Instance 0
    - N Body mass index (BMI) | Instance 0
    - N MET minutes per week for moderate activity | Instance 0
    - N MET minutes per week for vigorous activity | Instance 0
    - N MET minutes per week for walking | Instance 0
    - N Summed days activity | Instance 0
- lifestyle
    - C Alcohol drinker status | Instance 0
    - C Alcohol intake frequency. | Instance 0
    - C Exposure to tobacco smoke at home | Instance 0
    - C Exposure to tobacco smoke outside home | Instance 0
    - C Length of mobile phone use | Instance 0
    - C! Sleep duration | Instance 0
    - C Smoking status | Instance 0
    - C! Time spend outdoors in summer | Instance 0 
    - C! Time spent outdoors in winter | Instance 0
    - C! Time spent using computer | Instance 0
    - C! Time spent watching television (TV) | Instance 0
    - C! Weekly usage of mobile phone in last 3 months | Instance 0
- health
    - C Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor | Instance 0
    - C Cancer diagnosed by doctor | Instance 0
    - C Chest pain or discomfort | Instance 0
    - C Diabetes diagnosed by doctor | Instance 0
    - C Fractured/broken bones in last 5 years | Instance 0
    - C Mouth/teeth dental problems | Instance 0
    - C Other serious medical condition/disability diagnosed by doctor | Instance 0
    - C Overall health rating | Instance 0
    - C Vascular/heart problems diagnosed by doctor | Instance 0
    - N Health score
- other
    - C Age started wearing glasses or contact lenses | Instance 0
    - C Breastfed as a baby | Instance 0
    - C Getting up in morning | Instance 0
    - C How are people in household related to participant | Instance 0
    - C! Number in household | Instance 0
    - C Wears glasses or contact lenses | Instance 0

Also, which of these variables were included in the previous study? We should also make it easy to load those variables. I envision having a function which quickly selects variables based on categories.