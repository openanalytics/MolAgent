"""
written after prolif for calculating a Protein-Ligand Interaction Fingerprint 
================================================================================
"""

import os
import warnings
from collections.abc import Sized
from functools import wraps
from typing import Literal, Optional, Tuple

import multiprocess as mp
import numpy as np
from rdkit import Chem
from tqdm.auto import tqdm

from automol.structurefeatures.GradFormer.utils_3d.base import _BASE_INTERACTIONS, _INTERACTIONS
from automol.structurefeatures.GradFormer.utils_3d.interactions import *
from collections import UserDict
from rdkit.Chem.rdMolTransforms import GetDihedralDeg
from rdkit.Chem import AllChem
import MDAnalysis as mda
from rdkit import Chem
from scipy.spatial import cKDTree
from automol.structurefeatures.GradFormer.utils_3d.residue import ResidueId , Molecule , get_residues_near_ligand
#from utils import to_dataframe
from automol.structurefeatures.GradFormer.utils_3d.protein_data import *
from automol.structurefeatures.GradFormer.utils_3d.rdkit_util import cal_atomwise_Delta_sasa ,set_atomwise_TPSA, cal_sasa,get_sasa_residue_wise


######################################
############################


def first_occurence(interaction):
    @wraps(interaction)
    def wrapped(*args, **kwargs):
        return next(interaction(*args, **kwargs), None)

    return wrapped


def all_occurences(interaction):
    @wraps(interaction)
    def wrapped(*args, **kwargs):
        return tuple(interaction(*args, **kwargs))

    return wrapped


class Get_interactions:
    """Class that list interaction b. two molecules


    Parameters
    ----------
    interactions : list
        List of names (str) of interaction classes as found in the
        :mod:`prolif.interactions` module.
    parameters : dict, optional
        New parameters for the interactions. Mapping between an interaction name and a
        dict of parameters as they appear in the interaction class.
    
    vicinity_cutoff : float
        Automatically restrict the analysis to residues within this range of the ligand.
        This parameter is ignored if the ``residues`` parameter of the ``run`` methods
        is set to anything other than ``None``.

    """

    def __init__(
        self,
        interactions=[
 "Hydrophobic",
    "HBAcceptor",
    "HBDonor",
    "XBAcceptor",
    "XBDonor",
    "Cationic",
    "Anionic",
    "CationPi",
    "PiCation",
       "PiStacking_FaceToFace",
    "PiStacking_EdgeToFace",
    #"FaceToFace",    #"EdgeToFace", #   "PiStacking",
    "MetalDonor",
    "MetalAcceptor",
    "VdWContact",
 
        ],
        parameters=None,
        count=False,
        vicinity_cutoff=6.0,
    ):
        self._set_interactions(interactions, parameters)
        self.vicinity_cutoff = vicinity_cutoff

    def _set_interactions(self, interactions, parameters):
        # read interactions to compute
        parameters = parameters or {}
        if interactions == "all":
            interactions = self.list_available()
        # sanity check
        self._check_valid_interactions(interactions, "interactions")
        self._check_valid_interactions(parameters, "parameters")
        # add interaction methods
        self.interactions = {}
        wrapper = first_occurence
        for name, interaction_cls in _INTERACTIONS.items():
            # create instance with custom parameters if available
            interaction = interaction_cls(**parameters.get(name, {}))
            setattr(self, name.lower(), interaction)
            if name in interactions:
                self.interactions[name] = wrapper(interaction)

    def _check_valid_interactions(self, interactions_iterable, varname):
        """Raises a NameError if an unknown interaction is given."""
        unsafe = set(interactions_iterable)
        unknown = unsafe.symmetric_difference(_INTERACTIONS.keys() ) & unsafe
        if unknown:
            raise NameError(
                f"Unknown interaction(s) in {varname!r}: {', '.join(unknown)}"
            )

    def __repr__(self):  # pragma: no cover
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_interactions} interactions: {list(self.interactions.keys())}"
        return f"<{name}: {params} at {id(self):#x}>"

    @staticmethod
    def list_available(show_hidden=False):
        """List interactions available to the Fingerprint class.

        Parameters
        ----------
        show_hidden : bool
            Show hidden classes (base classes meant to be inherited from to create
            custom interactions).
        """
        if show_hidden:
            return sorted(_BASE_INTERACTIONS) + sorted(_INTERACTIONS)
        return sorted(_INTERACTIONS)

    @property
    def n_interactions(self):
        return len(self.interactions)


    def metadata(self, res1, res2):
        """Generates a metadata dictionary for the interactions between two residues.

        Parameters
        ----------
        res1 : prolif.residue.Residue
            A residue, usually from a ligand
        res2 : prolif.residue.Residue
            A residue, usually from a protein

        Returns
        -------
        metadata : dict[str, tuple[dict, ...]]
            Dict containing tuples of metadata dictionaries indexed by interaction
            name. If a specific interaction is not present between residues, it is
            filtered out of the dictionary.

        """
        return {
            name: (metadata,)
            for name, interaction in self.interactions.items()
            if (metadata := interaction(res1, res2, metadata=True))
        }

    def generate(self, lig, prot, residues=None):
        """Generates the interaction  between 2 molecules

        Parameters
        ----------
        lig : prolif.molecule.Molecule
            Molecule for the ligand
        prot : prolif.molecule.Molecule
            Molecule for the protein
        residues : list or "all" or None
            A list of protein residues (:class:`str`, :class:`int` or
            :class:`~prolif.residue.ResidueId`) to take into account for
            the fingerprint extraction. If ``"all"``, all residues will be
            used. If ``None``, at each frame the

        """
        out = {}
        prot_residues = prot.residues if residues == "all" else residues
        for lresid, lres in lig.residues.items():
            if residues is None:
                prot_residues = get_residues_near_ligand(
                    lres, prot, self.vicinity_cutoff
                )
            for prot_key in prot_residues:
                pres = prot[prot_key]
                key = (lresid, pres.resid)
                interactions = self.metadata (lres, pres)
                if any(interactions):
                    out[key] = interactions
                    #print(lres ,pres.resid.chain, pres.resid.name, pres.resid.number,interactions)
        return out
    def generatewithin_protein_interactions(self,  pro , interactions=["HBAcceptor",    "HBDonor",    
                                                                       "XBAcceptor",    "XBDonor",    
                                                                       "Cationic",    "Anionic",    "CationPi","PiCation",
                                                                       "PiStacking_FaceToFace",    "PiStacking_EdgeToFace",
                                                                       "MetalDonor",    "MetalAcceptor" ]):
        """Generates the interaction  within protein

        prot : prolif.molecule.Molecule

        """
        interaction_class={ 'Hydrophobic':   'Hydrophobic', 
                           'VdWContact':'VdWContact',
                            'Anionic':'Anionic-Cationic',
                           'Cationic': 'Anionic-Cationic',
                           'CationPi':'Cation-Pi',
                           'PiCation':'Cation-Pi',
                           'HBAcceptor':'H-bonding',
                            'HBDonor': 'H-bonding',
                           'PiStacking_EdgeToFace':'PiStacking',
                           'PiStacking_FaceToFace':'PiStacking',
                           'MetalAcceptor':'Metal-Don-Accr',
                           'MetalDonor':'Metal-Don-Accr',
                           'XBAcceptor': 'Halogen_bonding' ,
                           'XBDonor' : 'Halogen_bonding'  }
        tree = cKDTree(pro.xyz)
        def get_neihb_residues(res,cutoff=self.vicinity_cutoff):
            ix = tree.query_ball_point(res.xyz, cutoff)
            ix = set([i for lst in ix for i in lst])
            resids = [ResidueId.from_atom(pro.GetAtomWithIdx(i)) for i in ix]
            return list(set(resids))
        out = {}
        for res1id, res1 in pro.residues.items():
            neihb_residues=get_neihb_residues(res1)
            for res2id in neihb_residues:
                if (res2id, res1id) in out :#or res1id == res2id:
                    continue
                res2 = pro.residues[res2id]
                key = (res1id, res2id)
                #interactions = self.metadata (res1, res2)
                inters ={}
                for name, interaction in self.interactions.items():
                    if name not in interactions:
                        continue
                    if (metadata := interaction(res1, res2, metadata=True)):
                        nn=interaction_class[name]
                        #inters[name]= (metadata,)
                        inters[nn]= (metadata,)
                if any(inters):
                    out[key] = inters
                    #print(lres ,pres.resid.chain, pres.resid.name, pres.resid.number,interactions)
        return out



##################################################
def get_pepdides_bonds(mol):
        """
        returns a dic like 
        ResidueId(PHE, 7, A), ResidueId(GLU, 8, A)): {'original_atomid': (20, 22),  'atomnames': ('C', 'N')}
        """
        atoms2resinf={}
        for a in mol.GetAtoms():
            mi = a.GetPDBResidueInfo()
            if mi:
                atoms2resinf[a.GetIdx()]={
                 "name":  mi.GetName(),
                'resname' : mi.GetResidueName(),
                'resnr' : mi.GetResidueNumber(),
                'chid' : mi.GetChainId()
                }
        pepditeds_bonds={}
        for b in mol.GetBonds():
            bid=b.GetBeginAtomIdx()
            eid=b.GetEndAtomIdx()
            if atoms2resinf[bid]['resnr'] != atoms2resinf[eid]['resnr']:
                k1=ResidueId(atoms2resinf[eid]['resname'],atoms2resinf[eid]['resnr'], atoms2resinf[eid]['chid'])
                k2=ResidueId(atoms2resinf[bid]['resname'],atoms2resinf[bid]['resnr'], atoms2resinf[bid]['chid'])
                key= (k1,k2)
                pepditeds_bonds[key]={"parent_indices":(eid,bid),
                                      'atomnames':(atoms2resinf[eid]['name'].strip(),atoms2resinf[bid]['name'].strip() )
                                     }
        return pepditeds_bonds

##############


####################################################################
###############################################
from rdkit.Chem.rdMolTransforms import GetDihedralDeg
def get_atoms_resi_chid_mapping(mol):
    """
    returns a dict of peptide bonds list, bbdihedral psi/phi, rotamers
    ResidueId(PHE, 7, A), ResidueId(GLU, 8, A)): {'original_atomid': (20, 22),  'atomnames': ('C', 'N')}
    (ResidueId(THR, 139, A), ResidueId(PRO, 140, A)): {'phi': -51.647849730300095,  'psi': 160.2065441760023}
    """
    atoms2resinf={}
    for a in mol.GetAtoms():
        mi = a.GetPDBResidueInfo()
        if mi:
            resname=mi.GetResidueName().strip() 
            if resname=='':resname= 'UNK'
            chid= mi.GetChainId().strip()
            if chid =='':chid= None
            
            atoms2resinf[a.GetIdx()]={
             "name":  mi.GetName().strip(),
            'resname' : resname,
            'resnr' : mi.GetResidueNumber() or 0,
            'chid' : chid
            }
    atomname_resinf2IDX= {}
    resnr2resname={}
    for idx in atoms2resinf:
        key="{}_{}_{}".format(atoms2resinf[idx]['chid'],atoms2resinf[idx]['resnr'],atoms2resinf[idx]["name"])
        atomname_resinf2IDX[key]=idx
        k2= "{}_{}".format(atoms2resinf[idx]['chid'],atoms2resinf[idx]['resnr'])
        if k2 not in resnr2resname:
            resnr2resname[k2]=atoms2resinf[idx]['resname']
    return atoms2resinf , atomname_resinf2IDX ,resnr2resname

#############################################


class protein_internal_coor:
    """

    """

    def __init__(
        self,
        mol        
        
    ):
        self.mol=mol
        self.conformer=self.mol.GetConformer(0)
        assert self.conformer
        self.atoms2resinf , self.atomname_resinf2IDX, self.resnr2resname= self.get_atoms_resi_ch_mapping()
        self.res_res_bonds = None
        self.res_res_noncovalent_bonds=None
        self.backbone_dihs = None
        self.rotamers = None
        self.sasa_residue_wise=None
    ########################
    def get_atoms_resi_ch_mapping(self):
        return get_atoms_resi_chid_mapping(self.mol)
    #######################
    def get_res_res_noncovalent_bonds(self):
        fp=Get_interactions(interactions=["HBDonor","HBAcceptor",   
                                          "PiStacking_FaceToFace","PiStacking_EdgeToFace",
                                          "Anionic",  "Cationic", "CationPi",     "PiCation",])
        self.res_res_noncovalent_bonds=fp.generatewithin_protein_interactions(self.mol)
    #######################
    def get_res_res_bonds(self):
        """
        returns a dict of peptide bonds list, bbdihedral psi/phi, rotamers
        ResidueId(PHE, 7, A), ResidueId(GLU, 8, A)): {'original_atomid': (20, 22),  'atomnames': ('C', 'N')}
        """
        #if self.res_res_bonds:          return self.res_res_bonds
        if not self.atoms2resinf or not self.atomname_resinf2IDX or not self.resnr2resname:
                self.atoms2resinf , self.atomname_resinf2IDX, self.resnr2resname= self.get_atoms_resi_ch_mapping()
        ############################## get a dict of bonds boetween residures(peptide bonds+ disulfide_bonds)    
        res_res_bonds={}
        for b in self.mol.GetBonds():
            bid=b.GetBeginAtomIdx()
            eid=b.GetEndAtomIdx()
            if self.atoms2resinf[bid]['resnr'] != self.atoms2resinf[eid]['resnr']:
                key= ( ResidueId(self.atoms2resinf[eid]['resname'],self.atoms2resinf[eid]['resnr'], self.atoms2resinf[eid]['chid']),
                      ResidueId(self.atoms2resinf[bid]['resname'],self.atoms2resinf[bid]['resnr'], self.atoms2resinf[bid]['chid'])
                     )
                atomnames=(self.atoms2resinf[eid]['name'].strip(),self.atoms2resinf[bid]['name'].strip())
                parent_indices=(eid,bid)
                if atomnames== ('N','C'): 
                    parent_indices=(bid,eid)
                    atomnames= ('C','N')        
                tt={'parent_indices':parent_indices,
                        'atomnames': atomnames
                    }
                bond_type='UNK'
                if atomnames== ('C','N'):  bond_type="peptide_bond"
                elif atomnames== ('SG','SG'): bond_type="disulfide_bond"
                else:
                    print("unrecognized bond b. residues ", key , tt  )
                    #continue
                #if bond_type:
                if key not in res_res_bonds:  
                    res_res_bonds[key]={bond_type:tt}
                else: 
                    res_res_bonds[key][bond_type]=tt
        self.res_res_bonds= res_res_bonds    
    #########################################
    def get_backbone_dihs(self):
        """
        returns a dict 
        (ResidueId(THR, 139, A), ResidueId(PRO, 140, A)): {'phi': -51.647849730300095,  'psi': 160.2065441760023}
        """
        if not self.res_res_bonds:
            self.get_res_res_bonds()
        ######################################### get backbone psi, phi
        assert self.conformer
        bbdihs={}
        for k in self.res_res_bonds:
            if "peptide_bond" not in self.res_res_bonds[k]:# ['atomnames']!=  ('C','N'):
                #bbdihs[k]=None
                continue
            else:
                ids=self.res_res_bonds[k]["peptide_bond"]['parent_indices']
                res_C= "{}_{}".format( k[0].chain, k[0].number )
                res_N= "{}_{}".format( k[1].chain, k[1].number )
                #####################
                # #https://foldit.fandom.com/wiki/Backbone_angle
                #https://www.researchgate.net/figure/llustration-of-backbone-dihedral-angles-When-placing-the-atoms-for-residue-i-we-have-to_fig2_236959888
                # ϕ phi = cmd.get_dihedral(residue_def_prev+' and name C',residue_def+' and name N',residue_def+' and name CA',residue_def+' and name C')
                # psi ψ  = cmd.get_dihedral(residue_def+' and name N',residue_def+' and name CA',residue_def+' and name C',residue_def_next+' and name N')
                #ϕ phi   C(i-1)===N(i)---CA(i)---C(i)
                try:
                    phi=GetDihedralDeg(self.conformer,ids[0],ids[1], self.atomname_resinf2IDX[res_N+"_CA"], self.atomname_resinf2IDX[res_N+"_C"])
                except:
                    phi=np.nan
                # psi ψ  N(i)---CA(i)---C(i)===N(i+1)
                try:
                    psi=GetDihedralDeg(self.conformer,self.atomname_resinf2IDX[res_C+"_N"],self.atomname_resinf2IDX[res_C+"_CA"],ids[0],ids[1])
                except:
                    psi=np.nan
                # omega  CA(i)---C(i)===N(i+1)---CA(i+1)
                #omega=GetDihedralDeg(self.conformer,self.atomname_resinf2IDX[res_C+"_CA"],ids[0],ids[1], self.atomname_resinf2IDX[res_N+"_CA"])
                #print(phi,psi ,omega)
                bbdihs[k]={'phi':phi,
                          'psi':psi
                          }
        self.backbone_dihs=bbdihs
    ###############################
    def get_rotamers(self):
        ###get rotamers Chis
        if not self.atoms2resinf or not self.atomname_resinf2IDX or not self.resnr2resname:
                self.atoms2resinf , self.atomname_resinf2IDX, self.resnr2resname= self.get_atoms_resi_ch_mapping()
        rotamers={}
        for k in self.resnr2resname:
            resname=self.resnr2resname[k]
            if resname in CHIS:
                #print(k,resnr2resname[k])
                tmpdd={}
                for i,chi_atom_names in enumerate(CHIS[resname]):
                    try:
                        aa=[k+"_{}".format(a) for a in chi_atom_names]
                        #print(i,resname, chi_atom_names, aa)
                        for a in aa:
                            if a not in self.atomname_resinf2IDX:
                                print('atom not found', a, resname, chi_atom_names, aa )
                                continue
                            else:
                                cc=GetDihedralDeg(self.conformer, self.atomname_resinf2IDX[aa[0]], 
                                                             self.atomname_resinf2IDX[aa[1]],
                                                            self.atomname_resinf2IDX[aa[2]],
                                                            self.atomname_resinf2IDX[aa[3]])
                                chk='CHI{}'.format(i+1)
                                tmpdd[chk]=cc
                                #print(chk, cc)
                    except:     
                        continue
                if len(tmpdd):
                    if len(k.split('_'))==2:
                        chid=k.split('_')[0]
                        resnr=int(k.split('_')[1])
                    else:
                        chid=None
                        resnr=int(k.split('_')[0])
                    res_k=  ResidueId(resname,resnr, chid )
                    rotamers[res_k]=tmpdd
        self.rotamers= rotamers
    #####################
    def get_sasa_residue_wise(self):
        sasas={}
        if not  self.mol.GetAtoms()[0].HasProp("SASA"):
            cal_sasa(self.mol, total_only=True)
        if not  self.mol.GetAtoms()[0].HasProp("TPSA"):
            set_atomwise_TPSA(self.mol)
        for atom in self.mol.GetAtoms(): 
            key=ResidueId.from_atom( atom)
            if key not in sasas: sasas[key]={"SASA":0.0,'Polar':0.0, 'Apolar':0.0, 'TPSA':0.0} 
            t= atom.GetProp('SASAClassName')
            sasas[key][t] += atom.GetDoubleProp("SASA")
            sasas[key]["SASA"] += atom.GetDoubleProp("SASA")
            sasas[key]["TPSA"] += atom.GetDoubleProp("TPSA")
        self.sasa_residue_wise= sasas
    ######################################################
    def get_3ddist_matrix(self, seletion='CA'):
        dist_mat=AllChem.Get3DDistanceMatrix(self.mol)
        if seletion=='all':  return dist_mat
        if seletion=='CA':
            CAs_IDX={}
            for k in self.resnr2resname:
                if k+"_CA" not in self.atomname_resinf2IDX:
                    #print(k+"_CA")
                    continue
                caid= self.atomname_resinf2IDX[k+"_CA"]
                resname=self.resnr2resname[k]
                if len(k.split('_'))==2:
                        chid=k.split('_')[0]
                        resnr=int(k.split('_')[1])
                else:
                    chid=None
                    resnr=int(k.split('_')[0])
                res_k=  ResidueId(resname,resnr, chid )
                CAs_IDX[res_k]=caid
        kid=[v for v in CAs_IDX.values()]
        return dist_mat[kid,:][:,kid]
    ######################################################
    def get_centriods_dists(self):
        from scipy.spatial import distance_matrix

        centriods=[]
        keys=[]
        for residue in self.mol:
            keys.append(residue.resid)
            centriods.append(np.mean(residue.xyz, axis=0))
        xyz=np.stack(centriods, axis=0)
        dm=distance_matrix(xyz,xyz)
        spars_dist={}
        for r in range(len(keys)):
            for c in range(r,len(keys)):
                spars_dist[(keys[r],keys[c])]=dm[r,c]
        return spars_dist 
        
#########################################################################################

import numpy as np
#from utils_3d.get_interactions import Get_interactions, get_pepdides_bonds, CHIS, protein_internal_coor
from  automol.structurefeatures.GradFormer.utils_3d.rdkit_util import cal_atomwise_Delta_sasa
fp=Get_interactions(interactions=[
            "Hydrophobic",
            "HBDonor",
            "HBAcceptor",
            #"PiStacking",
    "PiStacking_FaceToFace",
    "PiStacking_EdgeToFace",
            "Anionic",
            "Cationic",
            "CationPi",
            "PiCation",
            "VdWContact",
        ])
#fp.list_available()

allowable_noncovalent_bonding_features = {

 'Is-Hydrophobic':  [ 'Hydrophobic'],  
'Is-VdWContact':['VdWContact'],
'Is-Anionic-Cationic':['Anionic','Cationic'],
 'Is-Cation-Pi': ['CationPi', 'PiCation'],
 'Is-H-bonding':['HBAcceptor', 'HBDonor'],
#'Is-PiStacking':['PiStacking'], 
'Is-PiStacking_EdgeToFace':['PiStacking_EdgeToFace'],
'Is-PiStacking_FaceToFace':['PiStacking_FaceToFace'],
'Is-Metal-Don-Accr' :['MetalAcceptor', 'MetalDonor'],
'Is-Halogen_bonding' : [  'XBAcceptor', 'XBDonor']#                                                  
    
}
def noncovalent_bond_to_feature_vector(bond):
    """

    """
    bond_feature = [ ]
    for k ,v in allowable_noncovalent_bonding_features.items() :
        #bond_feature.append( len(  intersection(v, bonds) ) > 0)
        bond_feature.append( bond in v)
    return bond_feature

def get_noncovalent_bond_feature_dims():
    return len(allowable_noncovalent_bonding_features)*[2]

interactions_parameters_names=[ 'distance','DHA_angle',"AXD_angle", "XAR_angle", 
                               'plane_angle', 'normal_to_centroid_angle','intersect_radius', 
                               'intersect_distance','angle_Cation_Pi']
interaction_class={ 'Hydrophobic':   'Hydrophobic', 
                           'VdWContact':'VdWContact',
                            'Anionic':'Anionic-Cationic',
                           'Cationic': 'Anionic-Cationic',
                           'CationPi':'Cation-Pi',
                           'PiCation':'Cation-Pi',
                           'HBAcceptor':'H-bonding',
                            'HBDonor': 'H-bonding',
                           'PiStacking_EdgeToFace':'PiStacking',
                           'PiStacking_FaceToFace':'PiStacking',
                           'MetalAcceptor':'Metal-Don-Accr',
                           'MetalDonor':'Metal-Don-Accr',
                           'XBAcceptor': 'Halogen_bonding' ,
                           'XBDonor' : 'Halogen_bonding'  }
################################
################################
def interactions_2graph(ligand_mol, protein_mol):
    """
    Converts interactions to graph Data object
    :input: rdkit mol
    :return: graph object
    """
    ############
    cal_atomwise_Delta_sasa( protein_mol,ligand_mol)
    pro_sasa_dic=get_sasa_residue_wise(protein_mol)
    pro_delta_sasa_Polar=[]
    pro_delta_sasa_Apolar=[]
    lig_delta_sasa_Polar=[]
    lig_delta_sasa_Apolar=[]
    for atom in ligand_mol.GetAtoms():
        if atom.GetProp('SASAClassName')== 'Polar':
            lig_delta_sasa_Polar.append(atom.GetDoubleProp("Delta_SASA"))
            lig_delta_sasa_Apolar.append(0.) 
        else: 
            lig_delta_sasa_Apolar.append(atom.GetDoubleProp("Delta_SASA"))
            lig_delta_sasa_Polar.append(0)
    ################
    i=0
    Protein_resid_key2ID={}
    for residue in protein_mol:
        k=residue.resid
        Protein_resid_key2ID[k]=i
        pro_delta_sasa_Polar.append(pro_sasa_dic[k]["Delta_SASA"]['Polar'])
        pro_delta_sasa_Apolar.append(pro_sasa_dic[k]["Delta_SASA"]['Apolar'])
        i+=1
        
    ##############
    
    ######################### residues info
    interactions_type = []
    ID2_key={}
    interactions_parameters=[]
    parent_indices=[]
    interactions_One_hot_encoding=[]
    ####################
    res=fp.generate(ligand_mol, protein_mol)
    edges_list=[]
    i=0
    for k in res:
        #ID2_key[i]  =  k
        #bt=[k2 for k2 in res[k]]
        #interactions_list.append(noncovalent_bond_to_feature_vector(bt))
        for tt in res[k]:
            interactions_type.append(interaction_class[tt])
            interactions_One_hot_encoding.append(noncovalent_bond_to_feature_vector(tt))
            ID2_key[i]  =  k
            parent_indices.append(res[k][tt][0]['parent_indices'])
            resid=Protein_resid_key2ID[k[1]]
            ligid=res[k][tt][0]['parent_indices']['ligand'][0]
            edges_list.append((ligid, resid))
            intpar=len(interactions_parameters_names)*[np.nan]
            for ii in res[k][tt][0]:
                #print(res[k][tt][0])
                if ii in ['indices' , 'parent_indices' ]: 
                        continue
                if ii in interactions_parameters_names: 
                    idint=interactions_parameters_names.index(ii)
                    intpar[idint]= res[k][tt][0][ii]
                else:
                    print(tt, ii, 'is not a recoginzed parameter')
            interactions_parameters.append(intpar) 
            i+=1
    
    out = dict()
    out['edges_list']=np.array(edges_list, dtype = np.int64).T
    out['interactions_list'] = interactions_type#np.array(interactions_type)
    out['interactions_One_hot_encoding']=np.array(interactions_One_hot_encoding,np.int64)
    out['interactions_parameters'] = np.array( interactions_parameters, dtype = np.float64)
    out['interactionID2_key'] = ID2_key
    out['protein_delta_sasa_Polar'] =np.array( pro_delta_sasa_Polar, dtype = np.float64)
    out['protein_delta_sasa_Apolar'] =np.array( pro_delta_sasa_Apolar, dtype = np.float64)
    out['ligand_delta_sasa_Polar'] =np.array( lig_delta_sasa_Polar , dtype = np.float64)
    out['ligand_delta_sasa_Apolar'] =np.array( lig_delta_sasa_Apolar , dtype = np.float64)
    #out['parent_indices']=parent_indices
    #out['Protein_resid_key2ID']=Protein_resid_key2ID
    return out 

                
                
        
        

            