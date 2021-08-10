from typing import Dict, List, Set, Iterable, Tuple
import numpy as np
from enum import Enum
import neuron
import re

NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH = 1


class SectionType(Enum):
    BRANCH_LEAF = 0
    BRANCH = 1
    BRANCH_INTERSECTION = 2
    SOMA = 3

    def __str__(self):
        return self.name


class SynapseNode:
    """
    create synapse tree as a DAG tree
    """

    def __init__(self, id: [int, None], length_to_next_node: [float, int, None] = None,
                 next_node: ['SynapseNode', None] = None, data: [neuron.nrn.Segment, None] = None):
        """
        :param id: unique id
        :param length_to_next_node:
        :param next_node: next node in the tree(to the root)
        """
        self.id = id
        self.next_node = next_node
        self.prev_nodes = set()
        self.length_to_next_node = length_to_next_node
        if self.next_node is not None:
            self.next_node.prev_nodes.add(self)
        # self.data = data

    def set_next_node(self, next_node, length_to_next_node):
        self.length_to_next_node = length_to_next_node
        self.next_node = next_node
        self.next_node.prev_nodes.add(self)

    def find_all_leafs(self) -> List['SynapseNode']:
        """
        find all leafs in the tree
        :return: array of leafs
        """
        root = self.find_root()
        return root._find_children_leafs()

    def _find_children_leafs(self):
        leaves = []
        current_node = self
        if (len(current_node.prev_nodes) > 0):
            for children in current_node.prev_nodes:
                leaves.extend(children._find_children_leafs())
            return leaves

        else:
            return [self]

    def get_all_children_id(self):
        """
        :return: all the children of current node
        """
        if self.prev_nodes is None:
            return {self.id}

        nodes_id = set()
        for prev_node in self.prev_nodes:
            nodes_id.add(prev_node.id)
            nodes_id.update(prev_node.get_all_children_id())
        return nodes_id

    def find_nodes_in_distance(self, distance: [float, int]) -> Set['SynapseNode']:
        """
        find all node in a certain distance upstream and downstream.
        :param distance: the maximal distance from current node.
        :return: set of nodes in distance.
        """
        nodes_in_distance = set()
        if distance < 0:
            return nodes_in_distance
        nodes_in_distance.add(self)
        if distance == 0:
            return nodes_in_distance
        if self.prev_nodes is not None:
            for prev_node in self.prev_nodes:
                if distance - prev_node.length_to_next_node < 0:
                    continue
                nodes_in_distance.update(prev_node.find_nodes_in_distance(distance - prev_node.length_to_next_node))
        if self.next_node is not None and distance - self.length_to_next_node >= 0:
            nodes_in_distance.update(self.next_node.find_nodes_in_distance(distance - self.length_to_next_node))
        return nodes_in_distance

    def find_root(self) -> 'SynapseNode':
        """
        :return: the root of the tree.
        """
        current_node = self
        while current_node.next_node is not None:
            current_node = self.next_node
        return current_node

    def __repr__(self):
        s1, s2 = (str(self.next_node.id), str(self.length_to_next_node)) if self.next_node is not None else (
            str(None), str(None))
        return "id: %i  next: %s  length: %s" % (self.id, s1, s2)

    def iterate_over_node_in_branch(self) -> Iterable['SynapseNode']:
        """
        :return: all nodes in a single branch from the farthest from the soma to the closest.
        """
        highest_node_on_branch = self
        while len(highest_node_on_branch.prev_nodes) > 0:
            if len(highest_node_on_branch.prev_nodes) > 1:
                break
            highest_node_on_branch = next(iter(highest_node_on_branch.prev_nodes))

        current_node = highest_node_on_branch
        while current_node.next_node is not None and len(current_node.next_node.prev_nodes) == 1:
            yield current_node
            current_node = current_node.next_node
        yield current_node

    def iterate_over_non_branch_first_child(self) -> Iterable['SynapseNode']:
        """
        find all the the synapses children that are not on current synapse branch .
        :return: children
        """
        highest_node_on_branch = self
        while len(highest_node_on_branch.prev_nodes) > 0:
            if len(highest_node_on_branch.prev_nodes) > 1:
                break
            highest_node_on_branch = next(iter(highest_node_on_branch.prev_nodes))

        current_node = highest_node_on_branch
        for child in current_node.prev_nodes:
            yield child

    @staticmethod
    def build_graph(distance_from_soma: np.ndarray, conectivity_matrix: np.ndarray) -> List['SynapseNode']:
        """
        fractory that create synapse tree by parameters.
        :param distance_from_soma: the distance of current node from soma
        :param conectivity_matrix: the connectivity matrix of nodes(also possible: the distance between them)
        :return: histogram of nodes.
        """
        number_of_synapses = distance_from_soma.size
        nodes_hist = [SynapseNode(i) for i in range(number_of_synapses)]
        row_connectivity, col_connectivity = np.nonzero(conectivity_matrix)
        for row_node, col_node in zip(row_connectivity, col_connectivity):
            if row_node >= col_node:  # symetric matrix
                continue
            if distance_from_soma[row_node] >= distance_from_soma[col_node]:
                parent, child = col_node, row_node
            else:
                child, parent = col_node, row_node
            nodes_hist[child].set_next_node(nodes_hist[parent], conectivity_matrix[row_node, col_node])
        return nodes_hist


class SectionNode:
    ID = 0

    def __init__(self, branch_or_synapse_nodes: [List['SectionNode'], 'SynapseNode', 'SectionNode'],is_soma=False):
        self.id = str(SectionNode.ID)
        SectionNode.ID += 1
        self.prev_nodes = []
        self.depth = None
        self.synapse_nodes_dict: Dict[int, 'SynapseNode'] = {}
        if isinstance(branch_or_synapse_nodes, SynapseNode):
            for node in branch_or_synapse_nodes.iterate_over_node_in_branch():
                self.synapse_nodes_dict[node.id] = node
            self.type = SectionType.BRANCH_LEAF
            self.representative = min(self.synapse_nodes_dict.keys())
        elif isinstance(branch_or_synapse_nodes, list) and not is_soma:
            for branch in branch_or_synapse_nodes:
                branch.next_node = self
                self.prev_nodes.append(branch)
            self.prev_nodes = sorted(self.prev_nodes, key=lambda node: node.representative)
            self.representative = self.prev_nodes[0].representative
            self.type = SectionType.BRANCH_INTERSECTION
        else:  # if soma
            self.prev_nodes = sorted(self.prev_nodes, key=lambda node: node.representative)
            for branch in branch_or_synapse_nodes:
                branch.next_node = self
                self.prev_nodes.append(branch)
            self.prev_nodes = sorted(self.prev_nodes, key=lambda node: node.representative)
            self.representative = self.prev_nodes[0].representative
            self.type = SectionType.SOMA
        self.next_node = None

    def connect_child_branch(self, child_node: 'SectionNode'):
        assert self.type == SectionType.BRANCH_LEAF and child_node.type == SectionType.BRANCH_INTERSECTION, \
            "child of type %s cannot be connected to type of %s" % s(child_node.type, self.type)
        # swallow up branch class
        # if child_node.type == SectionType.BRANCH_INTERSECTION:
        #     child_node.next_node = self.next_node
        #     self.prev_nodes = child_node.prev_nodes
        # else:
        child_node.next_node = self
        self.prev_nodes.append(child_node)
        self.representative = min(self.representative, child_node.representative)
        self.type = SectionType.BRANCH

    def get_number_of_parameters_for_nn(self):
        if self.type == SectionType.BRANCH_LEAF:
            return len(self.synapse_nodes_dict)
        elif self.type == SectionType.BRANCH:
            return len(self.synapse_nodes_dict) + NUMBER_OF_PREVIUSE_SEGMENTS_IN_BRANCH
        elif self.type == SectionType.BRANCH_INTERSECTION:
            return len(self.prev_nodes)
        elif self.type == SectionType.SOMA:
            return len(self.prev_nodes)

    def get_prev_node_representative(self):
        return [prev.representative for prev in self.prev_nodes]

    def __repr__(self):
        s = ""
        if self.type != SectionType.BRANCH_INTERSECTION:
            s = str(self.synapse_nodes_dict.keys())
        return "%d -> %s %s \t%s" % (
            self.representative, (str(self.next_node.representative) if self.next_node is not None else None),
            str(self.type), s)

    @staticmethod
    def build_segment_tree_from_synapsnodes(root: 'SynapseNode'):
        root_branch = SectionNode(root)
        children_non_branch_nodes = list(root.iterate_over_non_branch_first_child())
        if len(children_non_branch_nodes) > 0:
            branches = []
            for child in children_non_branch_nodes:
                branches.append(SectionNode.build_segment_tree_from_synapsnodes(child))
            branch_intersection = SectionNode(branches)
            root_branch.connect_child_branch(branch_intersection)
        # SegmentNode.update_depth(root_branch)
        return SectionNode(root_branch)

    def __iter__(self):
        # find leafs
        leafs_arr = []
        stack = [self]
        order_stack = []
        representative_set = set()
        while len(stack) > 0:
            current_node = stack.pop(0)
            order_stack.append(current_node)
            if current_node.type == SectionType.BRANCH:
                current_node = current_node.prev_nodes[0]
                order_stack.append(current_node)

            if current_node.type == SectionType.BRANCH_LEAF:
                leafs_arr.append(current_node)
                representative_set.add(current_node.representative)
            else:
                for child in current_node.prev_nodes:
                    stack.append(child)
        for child in reversed(order_stack):
            yield child

    def squeeze_tree(self):
        pass  # todo: add program that compute minimal tree where we remove branches without synapses.

    def number_of_branches(self):
        stack = [self]
        counter = 0
        while (len(stack) > 0):
            counter += 1
            node = stack.pop(0)
            for child in node.prev_nodes:
                stack.append(child)
        return counter

    def find_soma(self):
        current_node = self
        while current_node.next_node and current_node.type != SectionType.SOMA:
            current_node = next_node
        return current_node

    def update_depth(self):
        soma = self.find_soma()
        soma.depth = 0
        for child in soma.prev_nodes:
            child._update_depth()

    def _update_depth(self):
        self.depth = self.next_node.depth + 1
        if not self.prev_nodes:
            return
        for child in self.prev_nodes:
            child._update_depth()

    def max_depth(self):
        stack = [self.find_soma()]
        max_depth = 0
        while len(stack) > 0:
            node = stack.pop(0)
            max_depth = max(node.depth, max_depth)
            if node.prev_nodes:
                stack.extend(node.prev_nodes)
        return max_depth

    def pretty_print(self):
        root = self.find_soma()
        root.update_depth()
        max_depth = root.max_depth()
        level_array = [[] for _ in range(max_depth + 1)]
        stack = [root]
        while (len(stack) > 0):
            cur_node = stack.pop(0)
            level_array[cur_node.depth].append((cur_node))
            if cur_node.prev_nodes:
                stack.extend(cur_node.prev_nodes)

        max_length_of_level = max(level_array, key=lambda x: len(x))
        print(max_length_of_level)
        string_matrix = [[] for _ in range(max_depth + 1)]
        for i, str_arr in enumerate(string_matrix):
            spacing = len(max_length_of_level) // (len(level_array[i]) + 1)
            max_str = max(level_array[i], key=lambda x: len(str(x.representative)))
            max_str = str(max_str.representative)
            added_spacing = len(" |_ ")
            counter = 0
            for current_node in level_array[i]:
                for _ in range(spacing - 1):
                    string_matrix[counter] += " " * len(max_str)
                    counter += 1
                cur_str = str(current_node.representative)
                add_gap = len(max_str) - len(cur_str)
                cur_str += " " * add_gap
                counter += 1
            while (counter == max_length_of_level):
                string_matrix[i] += " " * len(max_str)
                counter += 1
        for s in string_matrix:
            print("".join(s))


def build_graph(model: neuron.hoc.HocObject,
                segment_index_mapping: [Dict[neuron.nrn.Segment, int], List[neuron.nrn.Segment], None] = None) -> [None,
                                                                                                                   SectionNode]:
    if not segment_index_mapping:
        list_of_basal_sections = [model.dend[x] for x in range(len(model.dend))]
        list_of_apical_sections = [model.apic[x] for x in range(len(model.apic))]
        all_sections = list_of_basal_sections + list_of_apical_sections
        all_segments = []
        for k, section in enumerate(all_sections):
            for currSegment in section:
                all_segments.append(currSegment)
        segment_index_mapping = all_segments
    if isinstance(segment_index_mapping, list):
        segment_index_mapping = create_from_histogram_mapping(segment_index_mapping)

    root = model.soma[0]
    segments = []
    is_tree_none = True
    for child in root.children():
        if "axon" in child.name():
            continue
        sub_tree = _build_subtree(child, segment_index_mapping)
        if sub_tree:
            is_tree_none = False
            segments.append(sub_tree)
    return None if is_tree_none else SectionNode(segments,is_soma=True) # intresection_banch and then soma.


def _build_subtree(section: neuron.nrn.Section, segment_index_mapping: Dict[neuron.nrn.Segment, int]) -> [None,
                                                                                                          SectionNode]:
    intersection_segment = None

    # if it has multiple childrens.
    if hasattr(section, 'children') and len(section.children()) > 1:
        child_sections = []
        for child in section.children():
            sub_tree = _build_subtree(child, segment_index_mapping)
            if sub_tree:
                child_sections.append(sub_tree)
        if len(child_sections) > 0:
            intersection_segment = SectionNode(child_sections)

    if hasattr(section, 'nseg') and section.nseg > 0:

        # implement synapse branch
        segments = []
        prev_node: [None, SynapseNode] = None
        cur_branch: [None, SectionNode] = None
        for seg in section:
            next_segment_length = 0  # if we want to add this factor.
            # if len(seg.point_processes()) == 0:
            #     next_segment_length += 0  # if we want to add this factor.
            #     continue
            synaps_index = segment_index_mapping[seg]
            segments.append(
                SynapseNode(synaps_index))  # id = synapse_id*number_of_different_synapse_types+synpse index(n*c+i)
            # s.t %i= type //c=id
            if prev_node:
                prev_node.set_next_node(segments[-1], next_segment_length)
        if len(segments) > 0:
            cur_branch = SectionNode(segments[-1])
        if intersection_segment:
            if cur_branch:
                cur_branch.connect_child_branch(intersection_segment)
            else:
                cur_branch = intersection_segment
        return cur_branch

    else:  # No segments on this branch :\
        if (not hasattr(section, 'children')) or len(section.children()) == 0:  # dead end
            return
        elif len(section.children()) == 1:  # only tube, which means that there is no information on this section
            return _build_subtree(section.children()[0], segment_index_mapping)
        else:  # it is only an intersection_segment
            return intersection_segment


def create_from_histogram_mapping(segment_histogram: List[neuron.nrn.Segment]):
    return {seg: i for i, seg in enumerate(segment_histogram)}

# %%
# a = SynapseNode.build_graph(np.array([0, 1, 1, 3, 4, 3]), np.array([[0, 2, 1, 0, 0, 0],
#                                                                     [2, 0, 0, 1, 0, 0],
#                                                                     [1, 0, 0, 0, 0, 1],
#                                                                     [0, 1, 0, 0, 1, 0],
#                                                                     [0, 0, 0, 1, 0, 0],
#                                                                     [0, 0, 1, 0, 0, 0]]))
# tree = build_graph(L5PC, allSegments)
