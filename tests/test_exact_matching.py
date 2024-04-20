from mapillm.exact_matching import ExactMatchingTool
import pytest 

@pytest.fixture
def exact_match():
    return ExactMatchingTool()


def test_exact_match(exact_match):
    formula="LiCoO2"
    space_group = 12
    lattice="Monoclinic"
    density=4.7
    prop="band_gap"

    output = exact_match.query_material_property(formula, prop, one_only=False)
    assert len(output) == 9

    output = exact_match.query_material_property(formula, prop, one_only=True)
    assert len(output) == 1

    specific_output = exact_match.query_material_property(formula=formula, space_group_num=space_group, density=density, lattice=lattice, desired_prop=prop)
    assert len(specific_output) == 1

    specific_output = exact_match.query_material_property(formula=formula, space_group_num=400, density=density, lattice=lattice, desired_prop=prop)
    assert len(specific_output) == 0

def test_get_prop(exact_match):
    formula="LiCoO2"
    prop="band_gap"

    output = exact_match._get_prop(exact_match.query_material_property(formula, prop), prop)

    assert type(output) == list and len(output) == 1 and type(output[0]) == float

def test_get_best_matches(exact_match):
    formula="LiFePO4"
    desired_prop="band_gap"
    space_group_num=250
    density=4.87
    lattice="Cubic"

    neighbors = exact_match.get_best_matches(formula, desired_prop, space_group_num=space_group_num, density=density, lattice=lattice, k=1)
    assert len(neighbors) >= 1

def test_average_neighbors(exact_match):
    input_dict = {3: [1,2], 2:[5,1], 1:[0.5]}
    average_neighbor = round(exact_match.average_neighbors(input_dict), 2)
    assert average_neighbor == 1.95

def test_get_weighted_average_of_neighbors(exact_match):
    #should succeed
    arg_dict = {"formula":"LiFePO4", "desired_prop":"band_gap", "space_group_num":250, "density":4.87, "lattice":"Cubic"}

    avg_prop = exact_match.get_weighted_average_of_neighbors(arg_dict)
    assert type(avg_prop) == float

    #should fail
    arg_dict = {"desired_prop":"band_gap", "space_group_num":250, "density":4.87, "lattice":"Cubic"}

    with pytest.raises(ValueError):
        avg_prop = exact_match.get_weighted_average_of_neighbors(arg_dict)

