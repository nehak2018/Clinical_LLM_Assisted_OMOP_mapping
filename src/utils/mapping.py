from data.sample import ICD_TO_OMOP

def map_to_omop(icds):
    return [ICD_TO_OMOP[i] for i in icds if i in ICD_TO_OMOP]