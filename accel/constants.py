from pathlib import Path

# project files

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# data subsets

EMOTIONAL = {'Irritability | Instance 0',
             'Miserableness | Instance 0',
             'Mood swings | Instance 0',
             'Sensitivity / hurt feelings | Instance 0',
             'Worrier / anxious feelings | Instance 0',
             'Frequency of tiredness / lethargy in last 2 weeks | Instance 0'}

DIETARY = {'Cooked vegetable intake | Instance 0',
           'Fresh fruit intake | Instance 0',
           'Oily fish intake | Instance 0',
           'Salad / raw vegetable intake | Instance 0',
           'Salt added to food | Instance 0',
           'Tea intake | Instance 0',
           'Water intake | Instance 0'}

SOCIO = {'Average total household income before tax | Instance 0',
         'Attendance/disability/mobility allowance | Instance 0',
         'Number of vehicles in household | Instance 0',
         'Qualifications | Instance 0',
         'Type of accommodation lived in | Instance 0',
         'Crime score',
         'Education score',
         'Employment score',
         'Housing score',
         'Income score',
         'Index of Multiple Deprivation',
         'Living environment',
         'Townsend deprivation index at recruitment'}

DEMO = {'Age at recruitment',
        'Ethnic background | Instance 0',
        'Sex'}

PHYSICAL = {'Above moderate/vigorous recommendation | Instance 0',
            'Above moderate/vigorous/walking recommendation | Instance 0',
            'Duration of walks | Instance 0',
            'Frequency of stair climbing in last 4 weeks | Instance 0',
            'IPAQ activity group | Instance 0',
            'Body mass index (BMI) | Instance 0',
            'MET minutes per week for moderate activity | Instance 0',
            'MET minutes per week for vigorous activity | Instance 0',
            'MET minutes per week for walking | Instance 0',
            'Summed days activity | Instance 0'}

LIFESTYLE = {'Alcohol drinker status | Instance 0',
             'Alcohol intake frequency. | Instance 0',
             'Exposure to tobacco smoke at home | Instance 0',
             'Exposure to tobacco smoke outside home | Instance 0',
             'Length of mobile phone use | Instance 0',
             'Sleep duration | Instance 0',
             'Smoking status | Instance 0',
             'Time spend outdoors in summer | Instance 0',
             'Time spent outdoors in winter | Instance 0',
             'Time spent using computer | Instance 0',
             'Time spent watching television (TV) | Instance 0',
             'Weekly usage of mobile phone in last 3 months | Instance 0'}

HEALTH = {'Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor | Instance 0',
          'Cancer diagnosed by doctor | Instance 0',
          'Chest pain or discomfort | Instance 0',
          'Diabetes diagnosed by doctor | Instance 0',
          'Fractured/broken bones in last 5 years | Instance 0',
          'Mouth/teeth dental problems | Instance 0',
          'Other serious medical condition/disability diagnosed by doctor | Instance 0',
          'Overall health rating | Instance 0',
          'Vascular/heart problems diagnosed by doctor | Instance 0',
          'Health score'}

OTHER = {'Age started wearing glasses or contact lenses | Instance 0',
         'Breastfed as a baby | Instance 0',
         'Getting up in morning | Instance 0',
         'How are people in household related to participant | Instance 0',
         'Number in household | Instance 0',
         'Wears glasses or contact lenses | Instance 0'}

FROM_PAPER = {'Age at recruitment',
              'Body mass index (BMI) | Instance 0',
              'Time spent watching television (TV) | Instance 0',
              'Time spent using computer | Instance 0',
              'Sleep duration | Instance 0',
              'Above moderate/vigorous recommendation | Instance 0',
              'Above moderate/vigorous/walking recommendation | Instance 0'}
            