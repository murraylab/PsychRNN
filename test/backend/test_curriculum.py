import pytest
import tensorflow as tf
from psychrnn.tasks.perceptual_discrimination import PerceptualDiscrimination
from psychrnn.backend.curriculum import Curriculum

def test_get_generator_function_deprecation():
	pds = [PerceptualDiscrimination(dt = 10, tau = 100, T = 2000, N_batch = 50, coherence = .7 - i/5) for i in range(4)]
	curriculum = Curriculum(pds)
	with pytest.raises(DeprecationWarning) as excinfo:
		curriculum.get_generator_function()
	assert 'Curriculum.batch_generator' in str(excinfo.value)
