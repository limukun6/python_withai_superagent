import unittest

from superagent.ab_scheduler import _first_route_configuration


def catalogue(values):
    caps = {f'arm{i}':dict(id=f'arm{i}',route='tree',resource='cpu',expected_classifier_fits=1,
                          params={'depth':v,'other':'fixed'}) for i,v in enumerate(values)}
    return caps, [(cap,1) for cap in caps.values()]


class TestFirstConfiguration(unittest.TestCase):
    def test_only_one_route_and_one_executable_parameter(self):
        caps, options = catalogue([1, 3, 9])
        caps['other'] = dict(id='other',route='linear',params={'depth':3},resource='cpu',expected_classifier_fits=1)
        options.insert(1,(caps['other'],1))
        self.assertEqual(_first_route_configuration(options,caps,[])[0]['id'], 'arm1')
        self.assertIs(_first_route_configuration(options,caps,['tree']), options[0])
        self.assertIs(_first_route_configuration(options,None,[]), options[0])
        caps['arm1']['resource'] = 'different'
        self.assertIs(_first_route_configuration(options,caps,[]), options[0])

    def test_even_menu_uses_original_order_on_tie_and_only_feasible_values(self):
        caps, options = catalogue([1, 2, 4, 8])
        shuffled = [options[i] for i in (3,2,1,0)]
        self.assertEqual(_first_route_configuration(shuffled,caps,[])[0]['id'], 'arm2')
        self.assertEqual(_first_route_configuration([options[0],options[2]],caps,[])[0]['id'], 'arm2')
        self.assertIs(_first_route_configuration([options[3]],caps,[]), options[3])

    def test_multidimensional_unordered_invalid_or_short_menus_keep_the_same_action(self):
        for levels in ([1,2], [1,2,2], [True,2,3], [0,2,4], [1,float('nan'),4], [1,float('inf'),4], ['a','b','c']):
            with self.subTest(levels=levels):
                caps,options = catalogue(levels)
                self.assertIs(_first_route_configuration(options,caps,[]), options[0])
        caps,options = catalogue([1,2,4])
        caps['arm1']['params']['other'] = 'changed'
        self.assertIs(_first_route_configuration(options,caps,[]), options[0])
        del caps['arm1']['params']['other']
        self.assertIs(_first_route_configuration(options,caps,[]), options[0])
