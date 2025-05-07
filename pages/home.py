
import streamlit as st

def home_page():
    st.title("HOME PAGE")
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

    st.header("About Me")
    st.markdown("""
    **Samarth Satalinga Kittad**
 # Add your photo here
    st.image("# Add your photo here
    st.image("https://avatars.githubusercontent.com/u/your-github-id?v=4", width=150)
    A passionate developer and computer aided drug discovery enthusiast.  
    I created this app to make protein analysis accessible, interactive, and visually engaging for students, researchers, and anyone curious about structural biology!
    """)

home_page()
