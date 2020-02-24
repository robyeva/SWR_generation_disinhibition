'''
Auxiliary python functions used by the figure generation plots
'''

# Return population name:
def pop_name(x):
    if x == 0: return 'P'
    if x == 1: return 'B'
    if x == 2: return 'A'

# Colors:
def pop_color(name):
    if name == 'P': return '#ef3b53'
    if name == 'B': return '#3c3fef'
    if name == 'A': return '#0a9045'

# Return parameter name as math text string:
def param_name(param):
    if len(param) == 3:
        return '$' + param[0].upper() + '_{' + param[1:].upper() + '}$'
    if len(param) == 2:
        return '$' + param[0].lower() + '_{' + param[1].upper() + '}$'
    else:
        return param

# Return parameter units:
def param_units(param):
    if param[0].upper() == 'W': return '[pA$\cdot$s]'
    if param[0].upper() == 'K': return '[1/pA]'
    if param[0].upper() == 'T': return '[pA]'

# Return string 'I->J' from weight W_JI:
def connection_name(param):
    return param[2].upper() + r'$\rightarrow$' + param[1].upper()
