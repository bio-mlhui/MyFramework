from lnn import Model
from lnn.symbolic.logic import Formula
from lnn import Predicates

from lnn import And , Fact


# leaf node是proposition/predicate, parent node是operators(and, or[n-ary], not[unary], impliles[binary], iff[binary])
# 整个图可以plot出来
# inference是omni-directional (upward & download)

# upward inference operator的truth value
# downward inference operand的truth

# LNN的truth value \in [0, 1]
Smokes, Cancer = Predicates('Smokes', 'Cancer')
model = Model()
model.infer()