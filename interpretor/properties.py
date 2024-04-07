import peptides
import numpy as np
import pandas as pd
import ipdb

def cal_discriptors(secq):
	
	peptide_information = {
		'A':np.array([89.1, 0, -1, 0, 0.25, 1]),
		'R':np.array([174.2, 1, 1, 0, -1.8, 6.13]),
		'N':np.array([132.1, 0, 1, 0,-0.64, 2.95]),
		'D':np.array([133.1, -1, 1, 0, -0.72, 2.78]),
		'C':np.array([121.2, 0, 1, 0, 0.04, 2.43]),
		'Q':np.array([146.2, 0, 1, 0, -0.69, 3.95]),
		'E':np.array([147.1, -1, 1, 0, -0.62, 3.78]),
		'G':np.array([75.1, 0, -1, 0, 0.16, 0]),
		'H':np.array([155.2, 1, 1, 0, -0.4, 4.66]),
		'I':np.array([131.2, 0, -1, 0, 0.73, 4]),
		'L':np.array([131.2, 0, -1, 0, 0.53, 4]),
		'K':np.array([146.2, 1, 1, 0, -1.1, 4.77]),
		'M':np.array([149.2, 0, -1, 0, 0.26, 4.43]),
		'F':np.array([165.2, 0, -1, 1, 0.61, 5.89]),
		'P':np.array([115.1, 0, -1, 0, -0.07, 2.72]),
		'S':np.array([105.1, 0, 1, 0, -0.26, 1.6]),
		'T':np.array([119.1, 0, 1, 0, -0.18, 2.6]),
		'W':np.array([204.2, 0, -1, 1, 0.37, 8.08]),
		'Y':np.array([181.2, 0, 1, 1, 0.02, 6.47]),
		'V':np.array([117.2, 0, -1, 0, 0.54, 3])
	}

	sum_of_all = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	positive_charge = 0
	negative_charge = 0
	polar_number = 0
	unpolar_number = 0
	ph_number = 0
	hydrophobicity = 0
	van_der_Waals_volume = 0
	
	for amino_acid in secq:
		sum_of_all += peptide_information[amino_acid]
		if peptide_information[amino_acid][1] == 1:
			positive_charge += 1
		elif peptide_information[amino_acid][1] == -1:
			negative_charge += 1
		if peptide_information[amino_acid][2] == 1:
			polar_number += 1
		elif peptide_information[amino_acid][2] == -1:
			unpolar_number += 1
	ph_number = sum_of_all[3]
	
	charge_of_all = sum_of_all[1]
	
	van_der_Waals_volume = sum_of_all[5]/len(secq)
	
	pep_discriptor = {
	
	'charge of all':charge_of_all,
	'positive_charge':positive_charge,
	'negative_charge':negative_charge,
	'polar_number':polar_number,
	'unpolar_number':unpolar_number,
	'ph_number':ph_number,
	
	'vdW_volume':van_der_Waals_volume,
	}
	
	return pep_discriptor


def Properties(df):

    Props = []
    for pep in df['Sequence']:

        peptide = peptides.Peptide(pep)
        # ipdb.set_trace()
        properties = {
                'ID': df[df['Sequence']==pep]["ID"].values[0],
                "Sequence": pep,
                "aliphatic_index" : peptide.aliphatic_index(),
                "boman" : peptide.boman(),
                "net charge11" : peptide.charge(pH=11),
                "net charge7" : peptide.charge(pH=7),
                "net charge3" : peptide.charge(pH=3),
                "isoelectric_point" : peptide.isoelectric_point(),
                "instability_index": peptide.instability_index(),
                "hydrophobicity" : peptide.hydrophobicity(),
                "molecular_weight" : peptide.molecular_weight(),
                "hydrophobic_moment_helix" : peptide.hydrophobic_moment(angle=100),
                "hydrophobic_moment_sheet" : peptide.hydrophobic_moment(angle=160),
               
            
        }
        properties.update(cal_discriptors(peptide))
        properties.update({"label": df[df['Sequence']==pep]["label"].values[0],
                "Idx": df[df['Sequence']==pep]["Idx"].values[0]})
                
        Props.append(properties)
    # ipdb.set_trace()
    Prop_df = pd.DataFrame(Props)
    return Prop_df