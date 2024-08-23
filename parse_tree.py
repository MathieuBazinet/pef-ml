import graphviz
import os

class TreeParser:
    def __init__(self, tree, max_depth, file_name):
        """

        :param tree: La représentation en string de l'arbre
        :param max_depth: La profondeur maximale de l'arbre
        :param file_name: Le nom à donner au fichier pdf
        """
        self.tree_string = """digraph Tree {
        node [shape=box, style="filled, rounded", color="black", fontname="helvetica"] ;
        graph [ranksep=equally, splines=polyline] ;
        edge [fontname="helvetica"] ;
        rankdir=LR ;"""
        self.tree = tree
        self.root_number = 0
        self.max_depth = max_depth
        self.dictionary = {i:[] for i in range(max_depth+1)}
        self.parse()
        print(self.tree_string)
        graph = graphviz.Source(self.tree_string)
        graph.dpi = 500
        graph.size = "50,50!"
        graph.render(file_name)
        os.remove(file_name)

    def find_separation(self, string):
        pos = string.find("(")
        number = 1
        for i in range(pos+1, len(string)):
            if number == 0:
                return string[:i], string[i+2:]
            if string[i] == "(":
                number += 1
            elif string[i] == ")":
                number -= 1

    def recursive_call(self, string, parent, depth):
        self.root_number += 1
        if string[:4] == "Leaf":
            label = string[5:string.find(")")].replace(" ", "<br/>")
            self.tree_string += f'{self.root_number} [label=<{label}>, fillcolor="#f2c19d"];\n'
            self.tree_string += f'{parent} -> {self.root_number} ;\n'
            self.dictionary[self.max_depth].append(self.root_number)
            return
        else:
            label = string[string.find("(") + 1:string.find(",")].replace("<=", "&le;")
            self.tree_string += f'{self.root_number} [label=<{label}>, fillcolor="#f8decb"];\n'
            if parent == 0 and self.root_number == 1:
                self.tree_string += '0 -> 1 [labeldistance=4, labelangle=-10, headlabel="True"] ;\n'
            elif parent == 0 and self.root_number != 1:
                self.tree_string += f'0 -> {self.root_number} [labeldistance=1.5, labelangle=-45, headlabel="False"] ;\n'
            else:
                self.tree_string += f'{parent} -> {self.root_number} ;\n'
            left, right = self.find_separation(string[string.find(",")+2:-1])
            self.dictionary[depth].append(self.root_number)
            number = self.root_number
            self.recursive_call(left, number, depth + 1)
            self.recursive_call(right, number, depth + 1)

    def parse(self):
        tree = self.tree
        if tree == "":
            return
        if self.max_depth == 0:
            self.tree_string = self.tree_string + "0 " + tree[tree.find("(")+1:tree.find(")")]

        depth = 0
        label = tree[tree.find("(")+1:tree.find(",")].replace("<=", "&le;")
        self.tree_string += f'{self.root_number} [label=<{label}>, fillcolor="#f8decb"];\n'
        left, right = self.find_separation(tree[tree.find(",")+2:-1])
        self.dictionary[depth].append(self.root_number)
        number = self.root_number
        self.recursive_call(left, number, depth+1)
        self.recursive_call(right, number, depth+1)
        for i in range(self.max_depth+1):
            self.tree_string += "{rank=same "
            for k in range(len(self.dictionary[i])):
                self.tree_string += f"; {self.dictionary[i][k]}"
            self.tree_string += "} ;\n"
        self.tree_string += "}"


if __name__ == "__main__":
    #tree = "Node(Isoelectric_point <= 8.3560, Node(Mol_weight <= 517.6240, Node(Isoelectric_point <= 4.0500, Leaf(cost:0.0000 pred:8.9400), Node(L <= 0.5000, Leaf(cost:0.5100 pred:2.6400), Leaf(cost:0.0000 pred:5.5850))), Node(Isoelectric_point <= 5.7140, Node(GRAVY Score <= 0.9200, Leaf(cost:8.3300 pred:0.0000), Leaf(cost:0.0000 pred:2.2600)), Node(Mol_weight <= 4749.3970, Leaf(cost:4.3700 pred:1.7300), Leaf(cost:0.0000 pred:0.0000)))), Node(Mol_weight <= 344.4160, Leaf(cost:0.0000 pred:8.7900), Node(Isoelectric_point <= 8.7480, Leaf(cost:0.0000 pred:2.6550), Node(GRAVY Score <= -0.5500, Leaf(cost:0.0000 pred:3.1450), Leaf(cost:1.5600 pred:4.2700)))))"
    d = {1: 'Node(Isoelectric_point <= 8.3560, Node(Mol_weight <= 517.6240, Node(Isoelectric_point <= 4.0500, Leaf(cost:0.0000 pred:8.9400), Node(L <= 0.5000, Leaf(cost:0.5100 pred:2.6400), Leaf(cost:0.0000 pred:5.5850))), Node(Isoelectric_point <= 5.7140, Node(GRAVY Score <= 0.9200, Leaf(cost:8.3300 pred:0.0000), Leaf(cost:0.0000 pred:2.2600)), Node(Mol_weight <= 4749.3970, Leaf(cost:4.3700 pred:1.7300), Leaf(cost:0.0000 pred:0.0000)))), Node(Mol_weight <= 344.4160, Leaf(cost:0.0000 pred:8.7900), Node(Isoelectric_point <= 8.7480, Leaf(cost:0.0000 pred:2.6550), Node(GRAVY Score <= -0.5500, Leaf(cost:0.0000 pred:3.1450), Leaf(cost:1.5600 pred:4.2700)))))',
        2: 'Node(Isoelectric_point <= 8.3560, Node(Isoelectric_point <= 5.2130, Node(Mol_weight <= 333.3410, Leaf(cost:0.0000 pred:7.0050), Node(Mol_weight <= 528.5620, Leaf(cost:0.0000 pred:0.1750), Leaf(cost:0.6600 pred:0.0000))), Node(Mol_weight <= 2019.2970, Node(L <= 0.5000, Leaf(cost:3.8300 pred:0.8850), Leaf(cost:0.0000 pred:1.9650)), Node(Isoelectric_point <= 7.4670, Leaf(cost:2.2300 pred:0.0000), Leaf(cost:0.0000 pred:1.1500)))), Node(Mol_weight <= 344.4160, Leaf(cost:0.0000 pred:5.3700), Node(Isoelectric_point <= 8.7480, Node(Mol_weight <= 2119.4810, Leaf(cost:0.4400 pred:1.6200), Leaf(cost:0.0000 pred:1.9400)), Node(Isoelectric_point <= 9.7500, Leaf(cost:4.1800 pred:2.4850), Leaf(cost:4.0100 pred:3.6050)))))',
        3: 'Node(Isoelectric_point <= 8.3560, Node(Isoelectric_point <= 5.2130, Node(Mol_weight <= 333.3410, Leaf(cost:0.0000 pred:6.7050), Node(Mol_weight <= 528.5620, Leaf(cost:0.0000 pred:0.2400), Leaf(cost:0.7600 pred:0.0000))), Node(Mol_weight <= 2019.2970, Node(L <= 0.5000, Leaf(cost:4.0900 pred:1.0200), Leaf(cost:0.0000 pred:2.2400)), Node(Isoelectric_point <= 7.4670, Leaf(cost:2.8500 pred:0.0000), Leaf(cost:0.0000 pred:1.2750)))), Node(Mol_weight <= 344.4160, Leaf(cost:0.0000 pred:6.1100), Node(Isoelectric_point <= 8.7480, Node(Mol_weight <= 1308.5500, Leaf(cost:0.0000 pred:2.1850), Leaf(cost:0.0400 pred:1.5700)), Node(Mol_weight <= 552.6330, Leaf(cost:0.0000 pred:1.4000), Leaf(cost:7.8600 pred:2.9150)))))',
        4: 'Node(Mol_weight <= 547.6790, Node(Mol_weight <= 344.4160, Node(Mol_weight <= 335.4040, Node(Mol_weight <= 307.3500, Leaf(cost:0.8900 pred:3.8350), Leaf(cost:0.1600 pred:4.7300)), Leaf(cost:0.1500 pred:6.7850)), Node(Isoelectric_point <= 4.0500, Leaf(cost:0.9300 pred:2.7300), Node(Isoelectric_point <= 6.1020, Leaf(cost:4.8100 pred:3.1600), Leaf(cost:0.0000 pred:3.4200)))), Node(Mol_weight <= 2019.2970, Node(Mol_weight <= 1245.4480, Node(GRAVY Score <= -0.2400, Leaf(cost:9.8800 pred:1.7800), Leaf(cost:14.4700 pred:2.5150)), Node(GRAVY Score <= 0.2100, Leaf(cost:16.3200 pred:1.8800), Leaf(cost:4.9100 pred:1.3550))), Node(Mol_weight <= 4749.3970, Node(Isoelectric_point <= 5.7020, Leaf(cost:2.0400 pred:0.0000), Leaf(cost:20.3500 pred:1.4450)), Leaf(cost:0.0000 pred:0.0000))))',
        5: 'Node(Isoelectric_point <= 8.7480, Node(Isoelectric_point <= 5.2130, Node(Mol_weight <= 333.3410, Leaf(cost:0.0000 pred:5.3700), Node(Mol_weight <= 528.5620, Leaf(cost:0.0000 pred:0.2400), Leaf(cost:0.5700 pred:0.0000))), Node(Mol_weight <= 2019.2970, Node(L <= 0.5000, Leaf(cost:3.4200 pred:0.9250), Leaf(cost:0.0000 pred:1.9900)), Node(Isoelectric_point <= 7.4670, Leaf(cost:1.6500 pred:0.0000), Leaf(cost:0.2000 pred:1.2350)))), Node(Mol_weight <= 344.4160, Leaf(cost:0.0000 pred:4.3550), Node(GRAVY Score <= -0.5000, Leaf(cost:1.4200 pred:1.5350), Node(GRAVY Score <= -0.4900, Leaf(cost:0.0000 pred:3.6250), Leaf(cost:1.5800 pred:2.2700)))))'}
    for i in range(1,6):
        TreeParser(d[i], 4, f"mmit_experience_{i}")