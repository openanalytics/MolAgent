


####################################################
 
# Atoms for each side-chain angle for each residue χn
CHIS = {}
CHIS["ARG"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD" ],
                ["CB","CG","CD","NE" ],
                ["CG","CD","NE","CZ" ]
              ]
 
CHIS["ASN"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","OD1" ]
              ]
 
CHIS["ASP"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","OD1" ]
              ]
CHIS["CYS"] = [ ["N","CA","CB","SG" ]
              ]
 
CHIS["GLN"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD" ],
                ["CB","CG","CD","OE1"]
              ]
 
CHIS["GLU"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD" ],
                ["CB","CG","CD","OE1"]
              ]
 
CHIS["HIS"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","ND1"]
              ]
 
CHIS["ILE"] = [ ["N","CA","CB","CG1" ],
                ["CA","CB","CG1","CD1" ]
              ]
 
CHIS["LEU"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD1" ]
              ]
 
CHIS["LYS"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD" ],
                ["CB","CG","CD","CE"],
                ["CG","CD","CE","NZ"]
              ]
 
CHIS["MET"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","SD" ],
                ["CB","CG","SD","CE"]
              ]
 
CHIS["PHE"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD1" ]
              ]
 
CHIS["PRO"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD" ]
              ]
 
CHIS["SER"] = [ ["N","CA","CB","OG" ]
              ]
 
CHIS["THR"] = [ ["N","CA","CB","OG1" ]
              ]
 
CHIS["TRP"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD1"]
              ]
 
CHIS["TYR"] = [ ["N","CA","CB","CG" ],
                ["CA","CB","CG","CD1" ]
              ]
 
CHIS["VAL"] = [ ["N","CA","CB","CG1" ]
              ]


##################################
AAthree2onedict= {'VAL':'V',    
                  'ILE':'I',    
                  'LEU':'L',     
                  'GLU':'E',     
                  'GLN':'Q',     
                  'ASP':'D',     
                  'ASN':'N',     
                  'HIS':'H',     
                  'HID': 'H',      # Histidine with hydrogen on the delta nitrogen.
                  'HIE':'H',      #     \# Histidine with hydrogen on the epsilon nitrogen.
                  'HIP': 'H',      # Histidine with hydrogens on both nitrogens; this is positively charged.
                  'TRP':'W',     
                  'PHE':'F',     
                  'TYR':'Y',     
                  'ARG':'R',     
                  'LYS':'K',     
                  'SER':'S',     
                  'THR':'T',     
                  'MET':'M',     
                  'ALA':'A',     
                  'GLY':'G',     
                  'PRO':'P',     
                  'CYS':'C' }


expected_atom_numbers = {'ALA':5,       'ARG':11,    'ASN':8,    'ASP':8,    'CYS':6,  'GLY':4,    'GLN':9,    'GLU':9,    
                         'HIS':10,   
                         'ILE':8,    
                         'LEU':8,    
                         'LYS':9,    
                         'MET':8,    
                         'PHE':11,   
                         'PRO':7,    
                         'SER':6,    
                         'THR':7,    
                         'TRP':14,   
                         'TYR':12,   
                         'VAL':7    	}
                         
modifiedAA = {	"AAR" : "ARG" ,		 
		"ABA" : "ALA" ,		 
		"AIB" : "ALA" ,		 
		"ALC" : "ALA" ,		 
		"ALN" : "ALA" ,		 
		"ALY" : "LYS" ,		 
		"ASN" : "ASN" ,		 
		"CME" : "CYS" ,		 
		"CSD" : "CYS" ,		 
		"CSO" : "CYS" ,		 
		"IIL" : "ILE" ,		 
		"DAB" : "ALA" ,		 
		"DPP" : "ALA" ,		 
		"HPE" : "PHE" ,		 
		"HPH" : "PHE" ,		 
		"NLE" : "LEU" ,		 
		"NVA" : "VAL" ,		 
		"OCS" : "CYS" ,		 
		"ORN" : "ALA" ,		 
		"PCA" : "GLU" ,		 
		"PPN" : "PHE" ,		 
		"PTR" : "TYR" ,		 
		"SEP" : "SER" ,		 
		"SMC" : "CYS" ,		 
		"TYS" : "TYR" ,		 
		"YCM" : "CYS"}
total_AA_resisues_names= [k for k in AAthree2onedict ] +[k for k in modifiedAA ] 

Sasa_symbol_radius = {
    #elements that actually occur in the regular amino acids and nucleotides
	#https://github.com/mittinatten/freesasa/blob/master/src/classifier.c
    "H": 1.10,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
    "SE": 1.90,
    # some others, values pulled from gemmi elem.hpp 
    # Halogens 
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.83,
    "I": 1.98,
    # Alkali and Alkali Earth metals 
    "LI": 1.81,
    "BE": 1.53,
    "NA": 2.27,
    "MG": 1.73,
    "K": 2.75,
    "CA": 2.31,
    "RB": 3.03,
    "SR": 2.49,
    "CS": 3.43,
    "BA": 2.68,
    "FR": 3.48,
    "RA": 2.83,
    # Transition metals 
    "SC": 2.11,
    "TI": 1.95,
    "V": 1.06,
    "CR": 1.13,
    "MN": 1.19,
    "FE": 1.26,
    "CO": 1.13,
    "NI": 1.63,
    "CU": 1.40,
    "ZN": 1.39,
    "Y": 1.61,
    "ZR": 1.42,
    "NB": 1.33,
    "MO": 1.75,
    "TC": 2.00,
    "RU": 1.20,
    "RH": 1.22,
    "PD": 1.63,
    "AG": 1.72,
    "CD": 1.58,
    "HF": 1.40,
    "TA": 1.22,
    "W": 1.26,
    "RE": 1.30,
    "OS": 1.58,
    "IR": 1.22,
    "PT": 1.75,
    "AU": 1.66,
    "HG": 1.55,
    # Post-Transition metals 
    "AL": 1.84,
    "GA": 1.87,
    "IN": 1.93,
    "SN": 2.17,
    "TL": 1.96,
    "PB": 2.02,
    "BI": 2.07,
    "PO": 1.97,
    # Metalloid 
    "B": 1.92,
    "SI": 2.10,
    "GE": 2.11,
    "AS": 1.85,
    "SB": 2.06,
    "TE": 2.06,
    "AT": 2.02,
    # Noble gases 
    "HE": 1.40,
    "NE": 1.54,
    "AR": 1.88,
    "KR": 2.02,
    "XE": 2.16,
    "RN": 2.20,
    # Lanthanoids 
    "LA": 1.83,
    "CE": 1.86,
    "PR": 1.62,
    "ND": 1.79,
    "PM": 1.76,
    "SM": 1.74,
    "EU": 1.96,
    "GD": 1.69,
    "TB": 1.66,
    "DY": 1.63,
    "HO": 1.61,
    "ER": 1.59,
    "TM": 1.57,
    "YB": 1.54,
    "LU": 1.53,
    # Actinoids 
    "AC": 2.12,
    "TH": 1.84,
    "PA": 1.60,
    "U": 1.86,
    "NP": 1.71,
    "PU": 1.67,
    "AM": 1.66,
    "CM": 1.65,
    "BK": 1.64,
    "CF": 1.63,
    "ES": 1.62,
    "FM": 1.61,
    "MD": 1.60,
    "NO": 1.59,
    "LR": 1.58,
}
