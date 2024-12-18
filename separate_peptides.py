import pandas as pd
from utils import separate_peptides

peptides = ['AAF', 'AAFQKVVAGVANA', 'AKL','AKLSEL','AVL',
'AVLGL','DAVMGNPKVKAHGKKVLQSFSDGLKHLDNLKGTF','DEVGGEALGRL','EEKEAVLGL',
'ERM','ERMFLGF','ERMFLGFPTTKTYFPHF','FESF','FESFGDL','FESFGDLSNA','FESFGDLSNADAVMGNPKVKAHGKKVLQSFSDGLKHLDNLKGTF',
'FLGF','FLGFPTTKTYFPHF','FLGFPTTKTYFPHFNLSHGSDQVKAHGQKVADALTKAVGHLDDLPGAL','FNLSHGSDQVK',
'FQKVVAGVANALAHKYH','FRLLG','FRLLGNV','FRLLGNVI','FRLLGNVIV','GDLSNADAVMGNPKVKAHGKKVLQSFSDGLKHLDNLKGTF',
'GHLDDLPGALSALSDLHAHKL','GLWGKVNV','GLWGKVNVDEVGGEAL','GLWGKVNVDEVGGEALGRL','GTFAKLSELHCDQLHVDP','HAHKLRVDPVNF',
'HASL','HPDDFNPSVHASLDKFLANVSTV','HVDPENF','IVVV','IVVVL','KVVAGVANALAHKYH','LAAHHPDDFNPSVHASLDKF','LAHKYH',
'LANV','LANVSTVL','LANVSTVLTSKYR','LERM','LGF','LGFPTTKTYFPHF','LGFPTTKTYFPHFNLSHGSDQVKAHGQKVADALTKAVGHLDDLPGAL',
'LGR','LLV','LLVT','LLVV','LLVVYPWTQRF','LTKAVGHLDDLPGAL','LVTL','LVTLAAHHPDDFNPSVHASL',
'LVTLAAHHPDDFNPSVHASLDKF','LVVYPWTQRF','NLSHGSDQVKAHGQKVADALTKAVGHLDDLPGAL','NLSHGSDQVKAHGQKVADALTKAVGHLDDLPGALSA',
'NLSHGSDQVKAHGQKVADALTKAVGHLDDLPGALSAL','PTTKTYFPHF','PTTKTYFPHFNLSHGSDQVKAHGQKVADALTKAVGHLDDLPGAL',
'PTTKTYFPHFNLSHGSDQVKAHGQKVADALTKAVGHLDDLPGALSAL','QKVVAGVANA','QKVVAGVANALAHKYH','RLLGNVIV','SAADKANVKAA',
'SAL','SALSDLHAHKLRVDPVNF','SALSDLHAHKLRVDPVNFKL','SDGLKHLDNLKGT','SDL','SDLHAHKLRVDPVNF','SKYR','SNADAVMGNPKVKAH',
'SNADAVMGNPKVKAHG','STVLTSKYR','TSKYR','TVLTSKYR','VGGEAL','VGGEALGRL','VGGQAGAHGAEALERMFL','VGHLDDLPGALS','VHLSAEE',
'VLSAADKANVKAAWGKVGGQAGAHGAEA','VLSAADKANVKAAWGKVGGQAGAHGAEAL','VLSAADKANVKAAWGKVGGQAGAHGAEALERM',
'VLTSKYR','VSTVL','VTLAAHHPDDFNPSVHASL','VTLAAHHPDDFNPSVHASLDKF','VTLAAHHPDDFNPSVHASLDKFLANV',
'VTLAAHHPDDFNPSVHASLDKFLANVSTVL','VVVL','VVVLARRLGHDFNPN','VVYPWTQRF','WGKVNV','WGKVNVDE','WGKVNVDEVGGEALGRL']

def main(exp_utilisateur, depth):
    array = pd.read_csv("./csv_laurent.csv")
    clfs = pd.read_pickle("./mean_vals_trees")
    clf = clfs[5*depth + exp_utilisateur - 6]
    tree = clf.tree_
    dictionary = separate_peptides(array, tree)
    print(dictionary)
    liste = list(dictionary.keys())
    liste.sort()
    for k in liste:
      kprime = [peptides[p] for p in dictionary[k]]
      print(k, kprime)

if __name__ == "__main__":
    exp_utilisateur = int(input("Quelle est l'expérience que tu veux effectuer ? "))

    while (exp_utilisateur < 1 or exp_utilisateur > 5):
        exp_utilisateur = int(input("Valeur incorrecte. Quelle est la première expérience que tu veux comparer ? "))

    depth = int(input("Quelle est la profondeur de ton arbre? "))
    while depth <= 0:
        depth = int(input("La profondeur doit être une valeur positive. Quelle est la profondeur de ton arbre? "))
    main(exp_utilisateur, depth)