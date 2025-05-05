import streamlit as st
import requests
import py3Dmol
import stmol
from Bio.PDB import PDBParser
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from ramachandraw.parser import get_phi_psi
from ramachandraw.utils import fetch_pdb, plot
import matplotlib.pyplot as plt
import shutil
import subprocess
import os
import tempfile

# ----------------------
# App Configuration
# ----------------------
st.set_page_config(
    page_title="Protein Molecule Mosaic",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Helper Functions
# ----------------------
@st.cache_data
def fetch_pdb_data(pdb_id):
    """Fetch PDB data from RCSB with error handling"""
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
    elif has_polydentate_properties(residue):
        return 'polydentate'
    return 'monodentate'

def has_polydentate_properties(residue):
    return any(atom.name in ['OXT', 'ND1', 'NE2'] for atom in residue)

def extract_ligands(pdb_data):
    parser = PDBParser()
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

def analyze_hydrogen_bonds(pdb_data):
    with open("temp.pdb", "w") as f:
        f.write(pdb_data)
    u = mda.Universe("temp.pdb")
    hbonds = HydrogenBondAnalysis(
        universe=u,
        donors_sel="name N",
        hydrogens_sel="name H",
        acceptors_sel="name O",
        d_a_cutoff=3.5,
        d_h_a_angle_cutoff=150,
    )
    hbonds.run()
    return hbonds.count_by_time()

def predict_active_sites(pdb_data):
    parser = PDBParser()
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

def create_3d_view(pdb_data, style='cartoon', highlight_ligands=True):
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    if style == 'cartoon':
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    elif style == 'surface':
        view.setStyle({'cartoon': {'color':'white'}})
        view.addSurface(py3Dmol.SAS, {'opacity':0.7})
    elif style == 'sphere':
        view.setStyle({'sphere': {'colorscheme':'Jmol'}})
    if highlight_ligands:
        view.addStyle({'hetflag': True}, 
                     {'stick': {'colorscheme':'greenCarbon', 'radius':0.3}})
    view.zoomTo()
    return view

def generate_ramachandran_plot(pdb_id):
    pdb_data = fetch_pdb_data(pdb_id)
    if pdb_data is not None:
        ax = plot(fetch_pdb(pdb_id))
        fig = ax.figure
        return fig
    return None

def generate_ramachandran_plot_from_data(pdb_string):
    """Generate Ramachandran plot from PDB string (uploaded file)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w") as tmp:
        tmp.write(pdb_string)
        tmp_path = tmp.name

    phi_psi = get_phi_psi(tmp_path)
    fig, ax = plt.subplots(figsize=(5,5))
    if phi_psi:
        ax.plot([x[0] for x in phi_psi], [x[1] for x in phi_psi], 'o', markersize=2)
    ax.set_title("Ramachandran Plot")
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True)
    return fig

def ramachandran_region_analysis(phi_psi_list):
    """
    Returns the percentage of residues in favored, allowed, and outlier regions.
    """
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

    # Avoid double-counting favored as allowed
    allowed = allowed - favored
    outlier = total - favored - allowed

    return {
        "favored": 100 * favored / total if total else 0,
        "allowed": 100 * allowed / total if total else 0,
        "outlier": 100 * outlier / total if total else 0,
        "total": total
    }

# ----------------------
# Docking UI Function
# ----------------------
def docking_ui(pdb_data):
    st.subheader("Ligand Docking (AutoDock Vina)")
    if shutil.which("vina") is None:
        st.error("AutoDock Vina is not installed or not in PATH. Please install and add to PATH.")
        return

    ligand_file = st.file_uploader("Upload ligand (PDBQT)", type=["pdbqt"])
    st.markdown("#### Docking Box Parameters")
    center_x = st.number_input("Center X", value=0.0, format="%.2f")
    center_y = st.number_input("Center Y", value=0.0, format="%.2f")
    center_z = st.number_input("Center Z", value=0.0, format="%.2f")
    size_x = st.number_input("Size X (Å)", value=20.0, min_value=5.0, max_value=60.0, format="%.2f")
    size_y = st.number_input("Size Y (Å)", value=20.0, min_value=5.0, max_value=60.0, format="%.2f")
    size_z = st.number_input("Size Z (Å)", value=20.0, min_value=5.0, max_value=60.0, format="%.2f")

    if ligand_file and pdb_data and st.button("Run Docking"):
        with open("protein.pdb", "w") as f:
            f.write(pdb_data)
        os.system("obabel protein.pdb -O protein.pdbqt")
        with open("ligand.pdbqt", "wb") as f:
            f.write(ligand_file.read())
        vina_cmd = [
            "vina",
            "--receptor", "protein.pdbqt",
            "--ligand", "ligand.pdbqt",
            "--center_x", str(center_x),
            "--center_y", str(center_y),
            "--center_z", str(center_z),
            "--size_x", str(size_x),
            "--size_y", str(size_y),
            "--size_z", str(size_z),
            "--out", "docked.pdbqt"
        ]
        st.write("Running docking...")
        result = subprocess.run(vina_cmd, capture_output=True, text=True)
        st.text(result.stdout)
        if os.path.exists("docked.pdbqt"):
            docked = open("docked.pdbqt").read()
            stmol.showmol(create_3d_view(docked, style='sphere'), height=400)
            st.success("Docking complete! Showing docked pose.")

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
# Main App Logic
# ----------------------
def main():
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
            view = create_3d_view(
                pdb_data, 
                style=controls['render_style'],
                highlight_ligands=controls['show_ligands']
            )
            stmol.showmol(view, height=600, width=800)
            with st.expander("Ramachandran Plot"):
                ramachandran_fig = generate_ramachandran_plot(pdb_id)
                if pdb_id:
                    phi_psi = get_phi_psi(fetch_pdb(pdb_id))
                    # ... your analysis or stats here ...
                if ramachandran_fig is not None:
                    st.pyplot(ramachandran_fig)
                else:
                    st.warning("Unable to generate Ramachandran plot. Please check the PDB ID.")

            # Docking feature below Ramachandran plot
            with st.expander("Ligand Docking (AutoDock Vina)"):
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
                active_sites = [res for res in PDBParser()
                    .get_structure("temp", StringIO(pdb_data))
                    .get_residues() if res.get_resname() in ['HIS', 'ASP', 'GLU']]
                st.write(f"**Potential Active Sites:** {len(active_sites)}")
                st.write("Common catalytic residues highlighted")
            with st.expander("Flexibility Report"):
                st.plotly_chart(px.histogram(x=range(10), y=range(10), title="Residue Flexibility"))
                st.caption("Simulated flexibility data - integrate MD analysis here")
            with st.expander("Hydrogen Bond Analysis"):
                hbond_counts = analyze_hydrogen_bonds(pdb_data)
                total_hbonds = np.sum(hbond_counts)
                st.write(f"Total Hydrogen Bonds: {total_hbonds}")
                if total_hbonds > 0:
                    st.write(f"Counts per Frame: {hbond_counts}")
            with st.expander("Active Site Prediction"):
                active_sites = predict_active_sites(pdb_data)
                st.write(f"**Predicted Active Sites ({len(active_sites)} residues):**")
                for site in active_sites:
                    st.write(f"{site['resname']} Chain {site['chain']} Residue {site['resnum']}")
                st.info("Active sites are predicted based on common catalytic residues (HIS, ASP, GLU, SER, CYS, LYS, TYR, ARG).")
            with st.expander("Ligand Type Visualization"):
                fig = visualize_ligand_counts(ligands)
                st.plotly_chart(fig)

            with st.expander("Ramachandran Plot"):
                ramachandran_fig = None
                phi_psi = None

                if source == "pdbid" and pdb_id:
                    ramachandran_fig = generate_ramachandran_plot(pdb_id)
                    try:
                        phi_psi = get_phi_psi(fetch_pdb(pdb_id))
                    except Exception as e:
                        phi_psi = None
                        st.warning(f"Could not extract phi/psi: {e}")
                elif source == "upload" and pdb_data:
                    ramachandran_fig = generate_ramachandran_plot_from_data(pdb_data)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w") as tmp:
                        tmp.write(pdb_data)
                        tmp_path = tmp.name
                    try:
                        phi_psi = get_phi_psi(tmp_path)
                    except Exception as e:
                        phi_psi = None
                        st.warning(f"Could not extract phi/psi: {e}")

                if phi_psi and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in phi_psi):
                    stats = ramachandran_region_analysis(phi_psi)
                    st.markdown(f"""
                    **Ramachandran Plot Analysis**
                    - **Total residues:** {stats['total']}
                    - **Favored region:** {stats['favored']:.1f}%
                    - **Allowed region:** {stats['allowed']:.1f}%
                    - **Outlier region:** {stats['outlier']:.1f}%
                    """)
                elif phi_psi:
                    st.warning("Phi/Psi data is not in the expected format.")

                if ramachandran_fig is not None:
                    st.pyplot(ramachandran_fig)
                else:
                    st.warning("Unable to generate Ramachandran plot. Please check the PDB input.")

if __name__ == "__main__":
    main()
