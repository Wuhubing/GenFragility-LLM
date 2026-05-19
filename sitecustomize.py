import sys
import types
module = types.ModuleType('pyairports')
module.airports = types.ModuleType('pyairports.airports')
module.airports.AIRPORT_LIST = []
sys.modules['pyairports'] = module
sys.modules['pyairports.airports'] = module.airports
