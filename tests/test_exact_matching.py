from mapillm.mapi_tools import ExactMatchingTool

def test_exact_match():
    formula="LiCoO2"
    space_group = 12
    lattice="Monoclinic"
    density=4.7
    prop="band_gap"

    matching_tool = ExactMatchingTool()
    output = matching_tool.query_material_property(formula, prop, one_only=False)
    assert len(output) == 9

    matching_tool = ExactMatchingTool()
    output = matching_tool.query_material_property(formula, prop, one_only=True)
    assert len(output) == 1

    matching_tool = ExactMatchingTool()
    specific_output = matching_tool.query_material_property(formula=formula, space_group_num=space_group, density=density, lattice=lattice, desired_prop=prop)
    assert len(specific_output) == 1

    matching_tool = ExactMatchingTool()
    specific_output = matching_tool.query_material_property(formula=formula, space_group_num=400, density=density, lattice=lattice, desired_prop=prop)
    assert len(specific_output) == 0

def test_get_prop():
    formula="LiCoO2"
    prop="band_gap"

    matching_tool = ExactMatchingTool()
    output = matching_tool._get_prop(matching_tool.query_material_property(formula, prop), prop)

    assert type(output) == list and len(output) == 1 and type(output[0]) == float