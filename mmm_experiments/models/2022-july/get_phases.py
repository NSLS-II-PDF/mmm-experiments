from mp_api import MPRester
from pymatgen.io.cif import CifWriter

if __name__ == "__main__":
    with MPRester("iX88TXTq45ESWLJCSSiT4OI6yZbnNFpl") as mpr:
        docs = mpr.summary.search(
            chemsys="Cu-Ti",
            energy_above_hull=(0, 0.1),
            fields=["material_id", "composition_reduced", "structure"],
        )
    for doc in docs:
        s = doc.structure
        path = (
            f"./{str(doc.composition_reduced).replace(' ','')}_{s.get_space_group_info()[0].replace('/','')}.cif"
        )
        CifWriter(s, symprec=0.1).write_file(str(path))
        print(path)
