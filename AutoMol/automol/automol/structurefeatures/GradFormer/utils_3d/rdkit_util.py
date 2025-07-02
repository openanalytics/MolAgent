#https://github.com/jaechanglim/DTI_PDBbind/blob/master/dataset.py#L22
from rdkit.Chem.TorsionFingerprints import CalculateTorsionLists, CalculateTorsionAngles
from rdkit.Chem.rdForceFieldHelpers import GetUFFVdWParams ,GetUFFTorsionParams
from rdkit.Chem.rdmolops import GetDistanceMatrix
from rdkit.Chem.rdmolops import SplitMolByPDBResidues
from rdkit.Chem import rdFreeSASA
from rdkit.Chem.rdmolops import CombineMols, GetAdjacencyMatrix
from rdkit import Chem
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem import rdmolops
from scipy.spatial import distance_matrix
from automol.structurefeatures.GradFormer.utils_3d.residue import ResidueId , Molecule
import math
##################################
from automol.structurefeatures.GradFormer.utils_3d.protein_data import AAthree2onedict ,modifiedAA
def extract_none_std_amino_acid(m):
    aa = SplitMolByPDBResidues(m)
    res = {}
    for k , v in aa.items():
        if k in AAthree2onedict or k in res:
            continue
        res[k]= v
    return res
#################################################
from automol.structurefeatures.GradFormer.utils_3d.protein_data import Sasa_symbol_radius
def classifyAtoms(mol, polar_atoms=[7, 8, 15, 16]):
    # Taken from https://github.com/mittinatten/freesasa/blob/master/src/classifier.c
    #https://github.com/jaechanglim/DTI_PDBbind/blob/master/dataset.py
    symbol_radius = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80,
                     "S": 1.80, "SE": 1.90, "FE": 2.05,
                     "F": 1.47, "CL": 1.75, "BR": 1.83, "I": 1.98,
                     "LI": 1.81, "BE": 1.53, "B": 1.92,
                     "NA": 2.27, "MG": 1.74, "AL": 1.84, "SI": 2.10,
                     "K": 2.75, "CA": 2.31, "GA": 1.87, "GE": 2.11, "AS": 1.85,
                     "RB": 3.03, "SR": 2.49, "IN": 1.93, "SN": 2.17, "SB": 2.06, "TE": 2.06,
                     "MN": 2.05, "ZN":1.39}
    symbol_radius.update(Sasa_symbol_radius)

    radii = []
    for atom in mol.GetAtoms():
        # mark everything as apolar to start
        atom.SetProp("SASAClassName", "Apolar")
        if atom.GetAtomicNum() in polar_atoms:  # identify polar atoms and change their marking
            atom.SetProp("SASAClassName", "Polar")  # mark as polar
        elif atom.GetAtomicNum() == 1:
            if len(atom.GetBonds())>0:
                if atom.GetBonds()[0].GetOtherAtom(atom).GetAtomicNum() \
                        in polar_atoms:
                    atom.SetProp("SASAClassName", "Polar")  # mark as polar
        radii.append(symbol_radius[atom.GetSymbol().upper()])
    return (radii)
#################################################
def cal_sasa(m, total_only=True):
    #radii = rdFreeSASA.classifyAtoms(m)
    radii = classifyAtoms(m)
    #radii = rdFreeSASA.classifyAtoms(m)
    sasa=rdFreeSASA.CalcSASA(m, radii)
    if total_only:
        return sasa,
    sasaApolar = rdFreeSASA.CalcSASA(m, radii, query=rdFreeSASA.MakeFreeSasaAPolarAtomQuery() )
    sasapolar = rdFreeSASA.CalcSASA(m, radii, query=rdFreeSASA.MakeFreeSasaPolarAtomQuery() )
    #if (sasa- (sasaApolar+ sasapolar)) > 0.1:        print("sasa=",(sasa- (sasaApolar+ sasapolar)))
    return sasa, sasaApolar ,sasapolar
#################################################
def cal_Delta_sasa( protein_mol,ligand_mol , total_only=True):
    lig_sasa=cal_sasa(ligand_mol, total_only=total_only)
    pro_sasa=cal_sasa(protein_mol, total_only=total_only)
    complex_mol = CombineMols(protein_mol,ligand_mol)
    com_sasa=cal_sasa(complex_mol, total_only=total_only)
    if total_only:
        return  com_sasa[0] - (pro_sasa[0]+lig_sasa[0])
    return com_sasa[0]-(pro_sasa[0]+lig_sasa[0]) ,com_sasa[1]-(pro_sasa[1]+lig_sasa[1]) ,com_sasa[2]-(pro_sasa[2]+lig_sasa[2])
#################################################
def cal_atomwise_Delta_sasa( protein_mol,ligand_mol):
    #pro_res={},    lig_res={}
    lig_sasa=cal_sasa(ligand_mol, total_only=True)
    pro_sasa=cal_sasa(protein_mol, total_only=True)
    for atom in protein_mol.GetAtoms():     
        atom.SetUnsignedProp("pro_org_index", atom.GetIdx())
        atom.SetDoubleProp("free_SASA",float(atom.GetProp("SASA") ) ) 
    for atom in ligand_mol.GetAtoms():     
        atom.SetUnsignedProp("lig_org_index", atom.GetIdx())
        atom.SetDoubleProp("free_SASA",float(atom.GetProp("SASA")) )
    complex_mol = CombineMols(protein_mol,ligand_mol)
    com_sasa=cal_sasa(complex_mol, total_only=True)
    for atom in complex_mol.GetAtoms():
        dsa= atom.GetDoubleProp("SASA") - atom.GetDoubleProp("free_SASA")
        if atom.HasProp("pro_org_index"):
            org_id=atom.GetUnsignedProp("pro_org_index")
            protein_mol.GetAtoms()[org_id].SetDoubleProp("Delta_SASA",dsa )
            #pro_res[org_id]=dsa
        if atom.HasProp("lig_org_index"):
            org_id=atom.GetUnsignedProp("lig_org_index")
            ligand_mol.GetAtoms()[org_id].SetDoubleProp("Delta_SASA",dsa )
            #lig_res[org_id]=dsa
    #return protein_mol,ligand_mol#pro_res, lig_res
#################################################
def get_sasa_residue_wise(mol):
        mol = Molecule.from_rdkit(mol)
        sasas={}
        if not  mol.GetAtoms()[0].HasProp("SASA"):
            cal_sasa(mol, total_only=True)
        if not mol.GetAtoms()[0].HasProp("TPSA"):
            set_atomwise_TPSA(mol)
        for atom in mol.GetAtoms(): 
            key=ResidueId.from_atom( atom)
            if key not in sasas: sasas[key]={"SASA":0.0,'Polar':0.0, 'Apolar':0.0, 'TPSA':0.0, "Delta_SASA":{'Polar':0.,'Apolar':0.0 }} 
            t= atom.GetProp('SASAClassName')
            sasas[key][t] += atom.GetDoubleProp("SASA")
            sasas[key]["SASA"] += atom.GetDoubleProp("SASA")
            sasas[key]["TPSA"] += atom.GetDoubleProp("TPSA")
            if atom.HasProp("Delta_SASA"): 
                sasas[key]["Delta_SASA"][t]+= atom.GetDoubleProp("Delta_SASA")
        return sasas


############################################
def set_atomwise_TPSA(m):
    #https://github.com/rdkit/rdkit/discussions/5352
    aa=Chem.rdMolDescriptors._CalcTPSAContribs(m)
    atoms=m.GetAtoms()
    for i in range(len(atoms)):
        atoms[i].SetDoubleProp("TPSA",aa[i] )
#################################################
from rdkit.Chem import GetPeriodicTable, PeriodicTable
def get_lone_pairs(atom) :
    """Get the number of lone pairs for an atom.
    https://github.com/rdkit/blob/master/Code/GraphMol/Aromaticity.cpp with
    github.com/AstraZeneca/jazzy/blob/master/src/jazzy/core.py
    """
    # set up a periodic table
    try:
        pt = Chem.GetPeriodicTable()
        symbol = atom.GetSymbol()
        valence_electrons = PeriodicTable.GetNOuterElecs(pt, symbol)
        unavailable_electrons = atom.GetExplicitValence()
        charge = atom.GetFormalCharge()
        free_electrons = valence_electrons - unavailable_electrons - charge
        return int(free_electrons / 2)
    except:
        return 0



#################################################

def cal_torsion_energy(m):
    energy = 0
    torsion_list, torsion_list_ring = CalculateTorsionLists(m)
    angles = CalculateTorsionAngles(m, torsion_list, torsion_list_ring)
    for idx, t in enumerate(torsion_list):
        indice, _ = t
        indice, angle = indice[0], angles[idx][0][0]
        v = GetUFFTorsionParams(m, indice[0], indice[1],
                                                    indice[2], indice[3])
        hs = [str(m.GetAtomWithIdx(i).GetHybridization()) for i in indice]
        if set([hs[1], hs[2]]) == set(["SP3", "SP3"]):
            n, pi_zero = 3, math.pi
        elif set([hs[1], hs[2]]) == set(["SP2", "SP3"]):
            n, pi_zero = 6, 0.0
        else:
            continue
        energy += 0.5 * v * (1 - math.cos(n * pi_zero) *
                             math.cos(n * angle / 180 * math.pi))
    return energy
#########################
def get_epsilon_sigma(m1, m2, mmff=True):
    if mmff:
        try:
            return get_epsilon_sigma_mmff(m1, m2)
        except:
            return get_epsilon_sigma_uff(m1, m2)
    return get_epsilon_sigma_uff(m1, m2)


def get_epsilon_sigma_uff(m1, m2):
    n1 = m1.GetNumAtoms()
    n2 = m2.GetNumAtoms()
    vdw_epsilon, vdw_sigma = np.zeros((n1, n2)), np.zeros((n1, n2))
    m_combine = CombineMols(m1, m2)
    for i1 in range(n1):
        for i2 in range(n2):
            param = GetUFFVdWParams(m_combine, i1, i1 + i2)
            if param is None:
                continue
            d, e = param
            vdw_epsilon[i1, i2] = e
            vdw_sigma[i1, i2] = d
            # print (i1, i2, e, d)
    return vdw_epsilon, vdw_sigma


def get_epsilon_sigma_mmff(m1, m2):
    n1 = m1.GetNumAtoms()
    n2 = m2.GetNumAtoms()
    vdw_epsilon, vdw_sigma = np.zeros((n1, n2)), np.zeros((n1, n2))
    m_combine = CombineMols(m1, m2)
    mp = ChemicalForceFields.MMFFGetMoleculeProperties(m_combine)
    for i1 in range(n1):
        for i2 in range(n2):
            param = mp.GetMMFFVdWParams(i1, i1 + i2)
            if param is None:
                continue
            d, e, _, _ = param
            vdw_epsilon[i1, i2] = e
            vdw_sigma[i1, i2] = d
            # print (i1, i2, e, d)
    return vdw_epsilon, vdw_sigma





########################
def cal_internal_vdw(m):
    retval = 0
    n = m.GetNumAtoms()
    c = m.GetConformers()[0]
    d = np.array(c.GetPositions())
    dm = distance_matrix(d, d)
    adj = GetAdjacencyMatrix(m)
    topological_dm = GetDistanceMatrix(m)
    for i1 in range(n):
        for i2 in range(0, i1):
            param = GetUFFVdWParams(m, i1, i2)
            if param is None:
                continue
            d, e = param
            d = d * 1.0
            if adj[i1, i2] == 1:
                continue
            if topological_dm[i1, i2] < 4:
                continue
            retval += e * ((d / dm[i1, i2]) ** 12 -
                           2 * ((d / dm[i1, i2]) ** 6))
            # print (i1, i2, e, d)
    return retval



###################################
#https://github.com/SCM-NV/PLAMS/blob/master/interfaces/molecule/rdkit.py
def optimize_coordinates(rdkit_mol, forcefield, fixed=[]):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def MMFFminimize():
        ff = AllChem.MMFFGetMoleculeForceField(rdkit_mol, AllChem.MMFFGetMoleculeProperties(rdkit_mol))
        for f in fixed:
            ff.AddFixedPoint(f)
        try:
            ff.Minimize()
        except:
            warn("MMFF geometry optimization failed for molecule: " + Chem.MolToSmiles(rdkit_mol))

    def UFFminimize():
        ff = AllChem.UFFGetMoleculeForceField(rdkit_mol, ignoreInterfragInteractions=True)
        for f in fixed:
            ff.AddFixedPoint(f)
        try:
            ff.Minimize()
        except:
            warn("UFF geometry optimization failed for molecule: " + Chem.MolToSmiles(rdkit_mol))

    optimize_molecule = {"uff": UFFminimize, "mmff": MMFFminimize}[forcefield]
    Chem.SanitizeMol(rdkit_mol)
    optimize_molecule()
    return
