import unittest
from analysis import checks
class DynamicsChecks(unittest.TestCase):
 def test_analytic_jacobian_and_refinement(self):checks()
if __name__=='__main__':unittest.main()
