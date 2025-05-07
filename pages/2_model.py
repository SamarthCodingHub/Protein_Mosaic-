import streamlit as st
import requests
from Bio.PDB import PDBParser, PPBuilder
from io import StringIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import py3Dmol

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
    # Use py3Dmol and st.components.v1.html
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
    # Render in Streamlit
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
# HOME PAGE
# ----------------------
def home_page():
    st.title("HOME PAGE")

    # Section 1: Overview of the App
    st.header("Overview of the App")
    st.markdown("""
    **Protein Molecule Mosaic** is an interactive web application for exploring and analyzing protein structures (PDB files).

    ### Objectives:
    - Make protein structure visualization accessible and interactive
    - Allow users to upload or fetch PDB files for analysis
    - Identify and classify ligands and predict active sites
    - Generate and analyze Ramachandran plots

    ### Features:
    - 3D visualization of protein structures (cartoon, surface, sphere)
    - Upload PDB files or fetch by PDB ID
    - Ligand classification (ion, monodentate, polydentate)
    - Active site prediction based on catalytic residues
    - Ramachandran plot generation with region analysis
    - User-friendly sidebar controls
    """)

    # Section 2: About Me
    st.header("About Me")
    st.markdown("""
    **Samarth Satalinga Kittad**

    A passionate developer and computer aided durg disocvery enthusiast.  
    I created this app to make protein analysis accessible, interactive, and visually engaging for students, researchers, and anyone curious about structural biology!
    """)


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
