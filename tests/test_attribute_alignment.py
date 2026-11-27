from unittest import TestCase

import numpy as np

from superagent.attribute_alignment import aligned_columns, comparable_fields, fitted_routes


class ConditionalAttributeTests(TestCase):
    def test_condition_and_fields_are_learned_after_column_reordering(self):
        rng=np.random.default_rng(9281)
        values=np.array(["elm","fern","plum","rose"])
        codes=rng.integers(0,4,(1400,3))
        phase=rng.integers(2,size=len(codes))
        kind=np.array(["oak","pine"])[phase]
        x=np.column_stack((values[codes],kind))
        y=(codes[np.arange(len(x)),phase]>=2).astype(int)
        route=fitted_routes(x,y,comparable_fields(x,[0,1,2]),[3])
        self.assertEqual(route[0][1],{"oak":0,"pine":1})
        shuffled=x[:,[2,3,0,1]]
        reroute=fitted_routes(shuffled,y,comparable_fields(shuffled,[0,2,3]),[1])
        np.testing.assert_array_equal(aligned_columns(x,route),aligned_columns(shuffled,reroute))
        novel=x[:6].copy()
        novel[:,3]="new"
        result=aligned_columns(novel,route)
        np.testing.assert_array_equal(result[:,0],novel[:,route[0][2]])
