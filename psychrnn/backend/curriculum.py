from __future__ import division
from __future__ import print_function

def default_metric(curriculum_params, input_data, correct_output, output_mask, output, epoch, losses, verbosity):
	accuracy = curriculum_params['accuracies'][curriculum_params['stage']](correct_output,output, output_mask)
	threshold = curriculum_params['thresholds'][curriculum_params['stage']]
	if verbosity:
		print("Accuracy: " + str(accuracy))
	return accuracy>threshold

class Curriculum(object):
	def __init__(self, tasks, **kwargs):
		self.stopTraining = False
		self.stage = 0
		self.tasks = tasks
		self.metric = kwargs.get('metric', default_metric)
		self.accuracies = kwargs.get('accuracy', [tasks[i].accuracy_function for i in range(len(tasks))])
		assert len(self.accuracies)==len(self.tasks)
		self.thresholds = kwargs.get('thresholds', [.9 for i in range(len(tasks))])
		assert len(self.thresholds)==len(self.tasks)
		self.metric_epoch = kwargs.get('metric_epoch', 10)
	
	def metric_test(self, input_data, correct_output, output_mask, test_output, epoch, losses, verbosity):
		if self.metric(self.__dict__, input_data, correct_output, output_mask, test_output, epoch, losses, verbosity):
			self.stage+=1
			if self.stage == len(self.tasks):
				self.stopTraining = True
			if verbosity:
				print("Stage " + str(self.stage))
			return True
		return False

	def get_generator_function(self):
		return self.tasks[self.stage].batch_generator()

	