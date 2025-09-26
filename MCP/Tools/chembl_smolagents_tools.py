"""
ChEMBL Tools for Smolagents
==========================

This module provides a comprehensive set of tools to integrate ChEMBL webresource client
functionality with the smolagents agentic system, enabling LLM-powered chemical data queries.

Install dependencies:
pip install chembl_webresource_client smolagents[toolkit]

Usage:
from smolagents import CodeAgent, InferenceClientModel
from chembl_smolagents_tools import get_all_chembl_tools

model = InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct")
chembl_tools = get_all_chembl_tools()
agent = CodeAgent(tools=chembl_tools, model=model, add_base_tools=True)

agent.run("Find molecules similar to aspirin with IC50 activities")
"""

import json
from typing import List, Dict, Any, Optional, Union
from smolagents import tool, Tool
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.utils import utils

MAX_NR_CPDs=200
class ChEMBLMoleculeSearchTool(Tool):
    """Advanced molecule search tool with multiple search criteria"""
    
    name = "chembl_molecule_search"
    description = "Search for molecules in ChEMBL database using various criteria like name, ChEMBL ID, molecular properties, etc."
    inputs = {
        "search_type": {
            "type": "string", 
            "description": "Type of search: 'chembl_id', 'name', 'synonym', 'inchi_key', 'molecular_weight', 'multiple_ids', 'rule_of_five'"
        },
        "query": {
            "type": "string", 
            "description": "Search query (ChEMBL ID, molecule name, synonym, InChI key, or MW threshold)"
        },
        "additional_filters": {
            "type": "string", 
            "description": " dictionary with additional filters like max_phase, biotherapeutic, etc."
        }
    }
    output_type = "string"
    
    def forward(self, search_type: str, query: str, additional_filters: str ) -> str:
        """Execute molecule search based on specified criteria"""
        try:
            molecule = new_client.molecule
            #filters = json.loads(additional_filters) if additional_filters else {}
            filters = additional_filters if additional_filters else {}
            
            if search_type == "chembl_id":
                results = molecule.filter(chembl_id=query)
            elif search_type == "name":
                results = molecule.filter(pref_name__iexact=query)
            elif search_type == "synonym":
                results = molecule.filter(molecule_synonyms__molecule_synonym__iexact=query)
            elif search_type == "inchi_key":
                results = molecule.filter(molecule_structures__standard_inchi_key=query)
            elif search_type == "molecular_weight":
                mw_threshold = float(query)
                results = molecule.filter(molecule_properties__mw_freebase__lte=mw_threshold)
            elif search_type == "multiple_ids":
                ids = [id.strip() for id in query.split(",")]
                results = molecule.filter(molecule_chembl_id__in=ids)
            elif search_type == "rule_of_five":
                violations = int(query) if query.isdigit() else 0
                results = molecule.filter(molecule_properties__num_ro5_violations=violations)
            else:
                return f"Error: Unsupported search type '{search_type}'"
            
            # Apply additional filters
            for key, value in filters.items():
                if key == "max_phase":
                    results = results.filter(max_phase=value)
                elif key == "biotherapeutic":
                    results = results.filter(biotherapeutic__isnull=False if value else True)
                elif key == "order_by":
                    results = results.order_by(value)
            
            # Limit results and format output
            limited_results = list(results.only(['molecule_chembl_id', 'pref_name', 'molecule_properties'])[:20])
            
            if not limited_results:
                return f"No molecules found for query: {query}"
            
            output = f"Found {len(limited_results)} molecules:\n"
            for mol in limited_results:
                props = mol.get('molecule_properties', {})
                mw = props.get('mw_freebase', 'N/A') if props else 'N/A'
                output += f"- {mol['molecule_chembl_id']}: {mol.get('pref_name', 'N/A')} (MW: {mw})\n"
            
            return output
            
        except Exception as e:
            return f"Error searching molecules: {str(e)}"


@tool
def chembl_activity_search(target_chembl_id: str, standard_type: str = "", assay_type: str = "", 
                          molecule_chembl_id: str = "", min_pchembl: float = 0.0) -> str:
    """
    Search for bioactivity data in ChEMBL database.
    
    Args:
        target_chembl_id: Target ChEMBL ID to search activities for
        standard_type: Activity standard type (e.g., 'IC50', 'Ki', 'EC50')
        assay_type: Assay type ('B' for binding, 'A' for ADMET, 'F' for functional)
        molecule_chembl_id: Optional specific molecule ChEMBL ID
        min_pchembl: Minimum pChEMBL value threshold
    """
    try:
        activity = new_client.activity
        
        # Build query filters
        query_filters = {"target_chembl_id": target_chembl_id}
        
        if standard_type:
            query_filters["standard_type"] = standard_type
        if assay_type:
            query_filters["assay_type"] = assay_type
        if molecule_chembl_id:
            query_filters["molecule_chembl_id"] = molecule_chembl_id
        if min_pchembl > 0:
            query_filters["pchembl_value__gte"] = min_pchembl
        
        results = activity.filter(**query_filters)
        
        # Limit results
        limited_results = list(results.only(['molecule_chembl_id', 'standard_type', 'standard_value', 
                                           'standard_units', 'pchembl_value', 'assay_description'])[:MAX_NR_CPDs])
        
        if not limited_results:
            return f"No activities found for target {target_chembl_id}"
        
        output = f"Found {len(limited_results)} activities for {target_chembl_id}:\n"
        for act in limited_results:
            value = act.get('standard_value', 'N/A')
            units = act.get('standard_units', '')
            pchembl = act.get('pchembl_value', 'N/A')
            output += f"- {act['molecule_chembl_id']}: {act['standard_type']} = {value} {units} (pChEMBL: {pchembl})\n"
        
        return output
        
    except Exception as e:
        return f"Error searching activities: {str(e)}"


@tool
def chembl_target_search(search_query: str, search_type: str = "name") -> str:
    """
    Search for targets in ChEMBL database.
    
    Args:
        search_query: Search term (target name, gene name, or synonym)
        search_type: Type of search ('name', 'gene', 'synonym', 'organism')
    """
    try:
        target = new_client.target
        
        if search_type == "name":
            results = target.filter(pref_name__icontains=search_query)
        elif search_type == "gene":
            results = target.filter(target_synonym__icontains=search_query)
        elif search_type == "synonym":
            results = target.filter(target_synonym__icontains=search_query)
        elif search_type == "organism":
            results = target.filter(organism__icontains=search_query)
        else:
            return f"Error: Unsupported search type '{search_type}'"
        
        limited_results = list(results.only(['target_chembl_id', 'pref_name', 'organism', 'target_type'])[:MAX_NR_CPDs])
        
        if not limited_results:
            return f"No targets found for query: {search_query}"
        
        output = f"Found {len(limited_results)} targets:\n"
        for tgt in limited_results:
            output += f"- {tgt['target_chembl_id']}: {tgt.get('pref_name', 'N/A')} ({tgt.get('organism', 'N/A')}) - {tgt.get('target_type', 'N/A')}\n"
        
        return output
        
    except Exception as e:
        return f"Error searching targets: {str(e)}"


@tool
def chembl_similarity_search(reference_molecule: str, similarity_threshold: int = 70, 
                           search_type: str = "chembl_id") -> str:
    """
    Find molecules similar to a reference molecule using Tanimoto similarity.
    
    Args:
        reference_molecule: Reference molecule (ChEMBL ID or SMILES)
        similarity_threshold: Minimum similarity threshold (0-100)
        search_type: Type of reference ('chembl_id' or 'smiles')
    """
    try:
        similarity = new_client.similarity
        
        if search_type == "chembl_id":
            results = similarity.filter(chembl_id=reference_molecule, similarity=similarity_threshold)
        elif search_type == "smiles":
            results = similarity.filter(smiles=reference_molecule, similarity=similarity_threshold)
        else:
            return f"Error: Unsupported search type '{search_type}'"
        
        limited_results = list(results.only(['molecule_chembl_id', 'pref_name', 'similarity'])[:MAX_NR_CPDs])
        
        if not limited_results:
            return f"No similar molecules found for {reference_molecule} with similarity >= {similarity_threshold}%"
        
        output = f"Found {len(limited_results)} similar molecules:\n"
        for mol in limited_results:
            output += f"- {mol['molecule_chembl_id']}: {mol.get('pref_name', 'N/A')} (Similarity: {mol['similarity']}%)\n"
        
        return output
        
    except Exception as e:
        return f"Error in similarity search: {str(e)}"


@tool
def chembl_drug_indication_search(indication_term: str) -> str:
    """
    Search for drugs by indication/disease.
    
    Args:
        indication_term: Disease or indication term to search for
    """
    try:
        drug_indication = new_client.drug_indication
        molecule = new_client.molecule
        
        # Find indications
        indications = drug_indication.filter(efo_term__icontains=indication_term)
        indication_list = list(indications[:MAX_NR_CPDs])
        
        if not indication_list:
            return f"No drug indications found for: {indication_term}"
        
        # Get molecules for these indications
        molecule_ids = [ind['molecule_chembl_id'] for ind in indication_list]
        molecules = molecule.filter(molecule_chembl_id__in=molecule_ids)
        molecule_list = list(molecules.only(['molecule_chembl_id', 'pref_name', 'max_phase'])[:MAX_NR_CPDs])
        
        output = f"Found {len(molecule_list)} drugs for indication '{indication_term}':\n"
        for mol in molecule_list:
            phase = mol.get('max_phase', 'N/A')
            output += f"- {mol['molecule_chembl_id']}: {mol.get('pref_name', 'N/A')} (Phase: {phase})\n"
        
        return output
        
    except Exception as e:
        return f"Error searching drug indications: {str(e)}"


@tool
def chembl_assay_search(description_term: str, assay_type: str = "") -> str:
    """
    Search for assays in ChEMBL database.
    
    Args:
        description_term: Term to search in assay descriptions
        assay_type: Optional assay type filter ('A', 'B', 'F', etc.)
    """
    try:
        assay = new_client.assay
        
        filters = {"description__icontains": description_term}
        if assay_type:
            filters["assay_type"] = assay_type
        
        results = assay.filter(**filters)
        limited_results = list(results.only(['assay_chembl_id', 'description', 'assay_type', 'assay_organism'])[:MAX_NR_CPDs])
        
        if not limited_results:
            return f"No assays found for description: {description_term}"
        
        output = f"Found {len(limited_results)} assays:\n"
        for assay_item in limited_results:
            desc = assay_item.get('description', 'N/A')[:100] + "..." if len(assay_item.get('description', '')) > 100 else assay_item.get('description', 'N/A')
            output += f"- {assay_item['assay_chembl_id']}: {desc} (Type: {assay_item.get('assay_type', 'N/A')})\n"
        
        return output
        
    except Exception as e:
        return f"Error searching assays: {str(e)}"


@tool
def chembl_molecule_image(chembl_id: str, image_format: str = "svg") -> str:
    """
    Get molecule structure image from ChEMBL.
    
    Args:
        chembl_id: ChEMBL ID of the molecule
        image_format: Image format ('svg' or 'png')
    """
    try:
        image = new_client.image
        image.set_format(image_format)
        
        image_data = image.get(chembl_id)
        
        if image_data:
            return f"Successfully retrieved {image_format.upper()} image for {chembl_id}. Image data length: {len(str(image_data))} characters."
        else:
            return f"No image found for molecule: {chembl_id}"
            
    except Exception as e:
        return f"Error retrieving molecule image: {str(e)}"


class ChEMBLMolecularUtilsTool(Tool):
    """Molecular utilities for SMILES processing and descriptor calculation"""
    
    name = "chembl_molecular_utils"
    description = "Perform molecular utility operations like SMILES to CTAB conversion, descriptor calculation, standardization, etc."
    inputs = {
        "operation": {
            "type": "string",
            "description": "Operation type: 'smiles_to_ctab', 'descriptors', 'standardize', 'parent_molecule', 'structural_alerts'"
        },
        "input_data": {
            "type": "string",
            "description": "Input molecule data (SMILES string or CTAB)"
        }
    }
    output_type = "string"
    
    def forward(self, operation: str, input_data: str) -> str:
        """Execute molecular utility operations"""
        try:
            if operation == "smiles_to_ctab":
                ctab = utils.smiles2ctab(input_data)
                return f"CTAB conversion successful. Length: {len(ctab)} characters."
            
            elif operation == "descriptors":
                # First convert SMILES to CTAB if needed
                if input_data.startswith("M  END") or "$$$$" in input_data:
                    ctab = input_data
                else:
                    ctab = utils.smiles2ctab(input_data)
                
                descriptors_json = utils.chemblDescriptors(ctab)
                descriptors = json.loads(descriptors_json)[0]
                
                output = "Molecular Descriptors:\n"
                for key, value in list(descriptors.items())[:MAX_NR_CPDs]:  # Limit to first 20 descriptors
                    output += f"- {key}: {value}\n"
                
                return output
            
            elif operation == "standardize":
                # Convert SMILES to CTAB if needed
                if input_data.startswith("M  END") or "$$$$" in input_data:
                    ctab = input_data
                else:
                    ctab = utils.smiles2ctab(input_data)
                
                standardized_json = utils.standardize(ctab)
                standardized = json.loads(standardized_json)
                
                return f"Standardization successful. Result: {standardized}"
            
            elif operation == "parent_molecule":
                # Convert SMILES to CTAB if needed
                if input_data.startswith("M  END") or "$$$$" in input_data:
                    ctab = input_data
                else:
                    ctab = utils.smiles2ctab(input_data)
                
                parent_json = utils.getParent(ctab)
                parent = json.loads(parent_json)
                
                return f"Parent molecule calculation successful. Result: {parent}"
            
            elif operation == "structural_alerts":
                # Convert SMILES to CTAB if needed
                if input_data.startswith("M  END") or "$$$$" in input_data:
                    ctab = input_data
                else:
                    ctab = utils.smiles2ctab(input_data)
                
                alerts_json = utils.structuralAlerts(ctab)
                alerts = json.loads(alerts_json)
                
                if alerts and alerts[0]:
                    output = "Structural Alerts Found:\n"
                    for alert in alerts[0]:
                        output += f"- {alert}\n"
                    return output
                else:
                    return "No structural alerts found."
            
            else:
                return f"Error: Unsupported operation '{operation}'"
                
        except Exception as e:
            return f"Error in molecular utils operation: {str(e)}"


@tool
def chembl_tissue_search(search_query: str, search_type: str = "name") -> str:
    """
    Search for tissues in ChEMBL database.
    
    Args:
        search_query: Search term
        search_type: Type of search ('name', 'uberon_id', 'bto_id', 'caloha_id')
    """
    try:
        tissue = new_client.tissue
        
        if search_type == "name":
            results = tissue.filter(pref_name__istartswith=search_query)
        elif search_type == "uberon_id":
            results = tissue.filter(uberon_id=search_query)
        elif search_type == "bto_id":
            results = tissue.filter(bto_id=search_query)
        elif search_type == "caloha_id":
            results = tissue.filter(caloha_id=search_query)
        else:
            return f"Error: Unsupported search type '{search_type}'"
        
        limited_results = list(results[:MAX_NR_CPDs])
        
        if not limited_results:
            return f"No tissues found for query: {search_query}"
        
        output = f"Found {len(limited_results)} tissues:\n"
        for tissue_item in limited_results:
            output += f"- {tissue_item.get('pref_name', 'N/A')} (Uberon: {tissue_item.get('uberon_id', 'N/A')})\n"
        
        return output
        
    except Exception as e:
        return f"Error searching tissues: {str(e)}"


@tool
def chembl_cell_line_search(search_query: str, search_type: str = "name") -> str:
    """
    Search for cell lines in ChEMBL database.
    
    Args:
        search_query: Search term
        search_type: Type of search ('name', 'cellosaurus_id')
    """
    try:
        cell_line = new_client.cell_line
        
        if search_type == "name":
            results = cell_line.filter(cell_name__icontains=search_query)
        elif search_type == "cellosaurus_id":
            results = cell_line.filter(cellosaurus_id=search_query)
        else:
            return f"Error: Unsupported search type '{search_type}'"
        
        limited_results = list(results[:MAX_NR_CPDs])
        
        if not limited_results:
            return f"No cell lines found for query: {search_query}"
        
        output = f"Found {len(limited_results)} cell lines:\n"
        for cell in limited_results:
            output += f"- {cell.get('cell_name', 'N/A')} (Cellosaurus: {cell.get('cellosaurus_id', 'N/A')})\n"
        
        return output
        
    except Exception as e:
        return f"Error searching cell lines: {str(e)}"


@tool
def chembl_document_search(search_query: str = "", doc_type: str = "", year_gte: int = 0) -> str:
    """
    Search for documents in ChEMBL database.
    
    Args:
        search_query: Search term for document title/content
        doc_type: Document type filter ('DATASET', 'PUBLICATION', etc.)
        year_gte: Minimum publication year
    """
    try:
        document = new_client.document
        
        filters = {}
        if search_query:
            filters["title__icontains"] = search_query
        if doc_type:
            filters["doc_type"] = doc_type
        if year_gte > 0:
            filters["year__gte"] = year_gte
        
        if not filters:
            return "Error: At least one search parameter must be provided"
        
        results = document.filter(**filters)
        limited_results = list(results.only(['document_chembl_id', 'title', 'doc_type', 'year', 'pubmed_id'])[:MAX_NR_CPDs])
        
        if not limited_results:
            return "No documents found matching the criteria"
        
        output = f"Found {len(limited_results)} documents:\n"
        for doc in limited_results:
            title = (doc.get('title', 'N/A')[:80] + "...") if len(doc.get('title', '')) > 80 else doc.get('title', 'N/A')
            output += f"- {doc['document_chembl_id']}: {title} ({doc.get('year', 'N/A')}, {doc.get('doc_type', 'N/A')})\n"
        
        return output
        
    except Exception as e:
        return f"Error searching documents: {str(e)}"


@tool
def chembl_list_available_resources() -> str:
    """
    List all available ChEMBL resources that can be queried.
    """
    try:
        available_resources = [resource for resource in dir(new_client) if not resource.startswith('_')]
        
        output = "Available ChEMBL Resources:\n"
        for resource in available_resources:
            output += f"- {resource}\n"
        
        return output
        
    except Exception as e:
        return f"Error listing resources: {str(e)}"


def get_all_chembl_tools() -> List[Tool]:
    """
    Get all ChEMBL tools for use with smolagents.
    
    Returns:
        List of ChEMBL tool instances
    """
    tools = [
        ChEMBLMoleculeSearchTool(),
        chembl_activity_search,
        chembl_target_search,
        chembl_similarity_search,
        chembl_drug_indication_search,
        chembl_assay_search,
        chembl_molecule_image,
        ChEMBLMolecularUtilsTool(),
        chembl_tissue_search,
        chembl_cell_line_search,
        chembl_document_search,
        chembl_list_available_resources
    ]
    
    return tools


