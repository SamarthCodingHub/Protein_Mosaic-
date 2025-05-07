import streamlit as st
import requests
from Bio.PDB import PDBParser, PPBuilder
from io import StringIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import py3Dmol

# Docking dependencies
import tempfile
import os
from vina import Vina
from meeko import MoleculePreparation
from rdkit import Chem
from rdkit.Chem import AllChem

# ----------------------
# Helper Functions
# ----------------------
@st.cache_data
def fetch_pdb_data(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Error fetching PDB data: {str(e)}")
        return None

def classify_ligand(residue):
    resname = residue.get_resname().strip()
    if len(resname) <= 2:
        return 'ion'
    elif any(atom.name in ['OXT', 'ND1', 'NE2'] for atom in residue):
        return 'polydentate'
    return 'monodentate'

def extract_ligands(pdb_data):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("temp", StringIO(pdb_data))
    ligands = {'ion': [], 'monodentate': [], 'polydentate': []}
    for residue in structure.get_residues():
        if residue.id[0] != ' ':
            ligand_type = classify_ligand(residue)
            if ligand_type == 'ion':
                ligands['ion'].append(residue.get_resname())
            else:
                ligands[ligand_type].append({
                    'resname': residue.get_resname(),
                    'chain': residue.parent.id,
                    'resnum': residue.id[1],
                    'type': ligand_type
                })
    return ligands

def predict_active_sites(pdb_data):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("temp", StringIO(pdb_data))
    catalytic_residues = ['HIS', 'ASP', 'GLU', 'SER', 'CYS', 'LYS', 'TYR', 'ARG']
    active_sites = []
    for residue in structure.get_residues():
        if residue.id[0] == ' ' and residue.get_resname() in catalytic_residues:
            active_sites.append({
                'resname': residue.get_resname(),
                'chain': residue.parent.id,
                'resnum': residue.id[1]
            })
    return active_sites

def visualize_ligand_counts(ligands):
    labels = list(ligands.keys())
    counts = [len(ligands[ligand_type]) for ligand_type in labels]
    fig = go.Figure(data=[
        go.Bar(name='Ligand Counts', x=labels, y=counts)
    ])
    fig.update_layout(title='Ligand Type Counts',
                      xaxis_title='Ligand Type',
                      yaxis_title='Count')
    return fig

def get_phi_psi_angles(pdb_string):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", StringIO(pdb_string))
    phi_psi = []
    for model in structure:
        for chain in model:
            ppb = PPBuilder()
            for pp in ppb.build_peptides(chain):
                angles = pp.get_phi_psi_list()
                for phi, psi in angles:
                    if phi is not None and psi is not None:
                        phi_psi.append((phi * 180.0 / 3.14159, psi * 180.0 / 3.14159))
    return phi_psi

def plot_ramachandran(phi_psi):
    fig, ax = plt.subplots(figsize=(5, 5))
    if phi_psi:
        phi, psi = zip(*phi_psi)
        ax.scatter(phi, psi, s=10)
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    ax.set_title("Ramachandran Plot")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True)
    return fig

def ramachandran_region_analysis(phi_psi_list):
    favored = 0
    allowed = 0
    outlier = 0
    total = len(phi_psi_list)
    for phi, psi in phi_psi_list:
        # Favored: α-helix region
        if -160 <= phi <= -40 and -80 <= psi <= -20:
            favored += 1
        # Favored: β-sheet region
        elif -180 <= phi <= -40 and 90 <= psi <= 180:
            favored += 1
        # Allowed (a generous margin)
        elif (-180 <= phi <= -20 and -180 <= psi <= 180):
            allowed += 1
        else:
            outlier += 1
    allowed = allowed - favored
    outlier = total - favored - allowed
    return {
        "favored": 100 * favored / total if total else 0,
        "allowed": 100 * allowed / total if total else 0,
        "outlier": 100 * outlier / total if total else 0,
        "total": total
    }

def show_3d_structure(pdb_data, style='cartoon', highlight_ligands=True):
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_data, 'pdb')
    if style == 'cartoon':
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    elif style == 'surface':
        view.setStyle({'cartoon': {'color': 'white'}})
        view.addSurface(py3Dmol.SAS, {'opacity': 0.7})
    elif style == 'sphere':
        view.setStyle({'sphere': {'colorscheme': 'Jmol'}})
    if highlight_ligands:
        view.addStyle({'hetflag': True}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.3}})
    view.zoomTo()
    view.setBackgroundColor('white')
    st.components.v1.html(view._make_html(), height=500, width=800)

# ----------------------
# Docking Helper Functions
# ----------------------
def smiles_to_pdbqt(smiles, out_pdbqt):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    mol_block = Chem.MolToMolBlock(mol)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mol') as tmp_mol:
        tmp_mol.write(mol_block.encode())
        tmp_mol.flush()
        preparator = MoleculePreparation()
        preparator.prepare(tmp_mol.name)
        with open(out_pdbqt, 'w') as f:
            f.write(preparator.write_pdbqt_string())
    os.unlink(tmp_mol.name)

def ligand_file_to_pdbqt(ligand_file, out_pdbqt):
    preparator = MoleculePreparation()
    preparator.prepare(ligand_file)
    with open(out_pdbqt, 'w') as f:
        f.write(preparator.write_pdbqt_string())

def protein_to_pdbqt(pdb_str, out_pdbqt):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as tmp_pdb:
        tmp_pdb.write(pdb_str.encode())
        tmp_pdb.flush()
        os.system(f'obabel {tmp_pdb.name} -O {out_pdbqt} -xr')
    os.unlink(tmp_pdb.name)

def run_vina_docking(receptor_pdbqt, ligand_pdbqt, center, box_size, exhaustiveness=8, n_poses=5):
    v = Vina(sf_name='vina')
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(center=center, box_size=box_size)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    poses = v.poses()
    scores = v.energies()
    docked_ligand_pdbqt = ligand_pdbqt + "_docked.pdbqt"
    v.write_poses(docked_ligand_pdbqt, n_poses=1, overwrite=True)
    return poses, scores, docked_ligand_pdbqt

def pdbqt_to_pdb(pdbqt_file):
    pdb_file = pdbqt_file + ".pdb"
    os.system(f'obabel {pdbqt_file} -O {pdb_file}')
    with open(pdb_file, 'r') as f:
        pdb_str = f.read()
    os.remove(pdb_file)
    return pdb_str

def show_docked_pose(protein_pdb, docked_ligand_pdbqt):
    ligand_pdb_str = pdbqt_to_pdb(docked_ligand_pdbqt)
    view = py3Dmol.view(width=800, height=500)
    view.addModel(protein_pdb, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.addModel(ligand_pdb_str, 'pdb')
    view.setStyle({'model': 1}, {'stick': {'colorscheme': 'greenCarbon'}})
    view.zoomTo()
    st.components.v1.html(view._make_html(), height=500, width=800)

# ----------------------
# UI Components
# ----------------------
def sidebar_controls():
    with st.sidebar:
        st.image("https://media.istockphoto.com/id/1390037416/photo/chain-of-amino-acid-or-bio-molecules-called-protein-3d-illustration.jpg?s=612x612&w=0&k=20&c=xSkGolb7TDjqibvINrQYJ_rqrh4RIIzKIj3iMj4bZqI=", width=400)
        st.title("Protein Molecule Mosaic")
        analysis_type = st.radio(
            "Analysis Mode:",
            ["Single Structure"],
            help="Analyze single structure"
        )
        render_style = st.selectbox(
            "Rendering Style:",
            ["cartoon", "surface", "sphere"],
            index=0,
            help="Choose molecular representation style"
        )
        st.markdown("---")
        st.markdown("**Ligand Display Options**")
        show_ligands = st.checkbox("Highlight Ligands", True)
        return {
            'analysis_type': analysis_type,
            'render_style': render_style,
            'show_ligands': show_ligands,
        }

# ----------------------
# Docking UI
# ----------------------
def docking_ui(pdb_data):
    st.header("Protein-Ligand Docking (AutoDock Vina)")
    st.markdown("Upload a ligand (SDF/MOL2) or enter a SMILES string to dock against the loaded protein.")

    ligand_file = st.file_uploader("Ligand file (SDF/MOL2)", type=["sdf", "mol2"])
    ligand_smiles = st.text_input("Or enter ligand SMILES")
    ligand_name = st.text_input("Ligand name (for output)", value="LIG")

    st.subheader("Docking Box Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        x = st.number_input("Center X", value=0.0)
    with col2:
        y = st.number_input("Center Y", value=0.0)
    with col3:
        z = st.number_input("Center Z", value=0.0)
    box_size = st.number_input("Box size (Å)", value=20.0)

    if st.button("Run Docking"):
        if not (ligand_file or ligand_smiles):
            st.error("Please provide a ligand.")
            return

        with st.spinner("Preparing files and running docking..."):
            # Prepare protein
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdbqt') as tmp_prot:
                protein_to_pdbqt(pdb_data, tmp_prot.name)
                protein_pdbqt = tmp_prot.name
            # Prepare ligand
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdbqt') as tmp_lig:
                if ligand_file:
                    ligand_file_to_pdbqt(ligand_file, tmp_lig.name)
                else:
                    smiles_to_pdbqt(ligand_smiles, tmp_lig.name)
                ligand_pdbqt = tmp_lig.name

            # Run docking
            poses, scores, docked_ligand_pdbqt = run_vina_docking(
                protein_pdbqt,
                ligand_pdbqt,
                center=[x, y, z],
                box_size=[box_size, box_size, box_size]
            )
            st.success("Docking completed!")
            st.write("Top docking scores (kcal/mol):")
            for i, score in enumerate(scores):
                st.write(f"Pose {i+1}: {score[0]:.2f}")

            # Visualize top pose
            st.subheader("Docked Pose (Top Scoring)")
            show_docked_pose(pdb_data, docked_ligand_pdbqt)

            # Clean up
            os.remove(protein_pdbqt)
            os.remove(ligand_pdbqt)
            os.remove(docked_ligand_pdbqt)

# ----------------------
# Main App Logic
# ----------------------
def main():
    st.set_page_config(
        page_title="Protein Molecule Mosaic",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    controls = sidebar_controls()
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("Protein Palette")
        st.markdown("**Load a protein structure:**")
        pdb_id = st.text_input("Enter PDB ID (optional):").upper()
        uploaded_pdb = st.file_uploader("Or upload a PDB file", type=["pdb"])
        pdb_data = None
        source = None
        if uploaded_pdb is not None:
            pdb_data = uploaded_pdb.read().decode("utf-8")
            source = "upload"
            st.success("PDB file uploaded and loaded.")
        elif pdb_id:
            pdb_data = fetch_pdb_data(pdb_id)
            source = "pdbid"
            if pdb_data:
                st.success(f"PDB ID {pdb_id} loaded from RCSB.")

        if pdb_data:
            st.subheader("3D Structure Viewer")
            show_3d_structure(
                pdb_data,
                style=controls['render_style'],
                highlight_ligands=controls['show_ligands']
            )
            with st.expander("Ramachandran Plot"):
                phi_psi = get_phi_psi_angles(pdb_data)
                if phi_psi:
                    fig = plot_ramachandran(phi_psi)
                    st.pyplot(fig)
                    stats = ramachandran_region_analysis(phi_psi)
                    st.markdown(f"""
                    **Ramachandran Plot Analysis**
                    - **Total residues:** {stats['total']}
                    - **Favored region:** {stats['favored']:.1f}%
                    - **Allowed region:** {stats['allowed']:.1f}%
                    - **Outlier region:** {stats['outlier']:.1f}%
                    """)
                else:
                    st.warning("Unable to generate Ramachandran plot. Please check the PDB input.")

            with st.expander("Protein-Ligand Docking"):
                docking_ui(pdb_data)

    with col2:
        st.header("Protein Dynamics")
        if pdb_data:
            with st.expander("Ligand Information"):
                ligands = extract_ligands(pdb_data)
                st.write(f"**Ions:** {len(ligands['ion'])}")
                st.write(f"**Ion Names:** {', '.join(ligands['ion'])}")
                st.write(f"**Monodentate Ligands:** {len(ligands['monodentate'])}")
                st.write(f"**Polydentate Ligands:** {len(ligands['polydentate'])}")
            with st.expander("Active Sites"):
                active_sites = predict_active_sites(pdb_data)
                st.write(f"**Predicted Active Sites ({len(active_sites)} residues):**")
                for site in active_sites:
                    st.write(f"{site['resname']} Chain {site['chain']} Residue {site['resnum']}")
                st.info("Active sites are predicted based on common catalytic residues (HIS, ASP, GLU, SER, CYS, LYS, TYR, ARG).")
            with st.expander("Ligand Type Visualization"):
                fig = visualize_ligand_counts(ligands)
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()

             
