from __future__ import division
from __future__ import print_function

from os import makedirs, path
import numpy as np

def default_metric(curriculum_params, input_data, correct_output, output_mask, output, epoch, losses, verbosity):
	accuracy = curriculum_params['accuracies'][curriculum_params['stage']](correct_output,output, output_mask)
	threshold = curriculum_params['thresholds'][curriculum_params['stage']]
	if verbosity:
		print("Accuracy: " + str(accuracy))
	return accuracy>threshold, accuracy

class Curriculum(object):
	def __init__(self, tasks, **kwargs):
		self.stopTraining = False
		self.stage = 0
		self.metric_values = []

		# List of tasks that make up the curriculum
		self.tasks = tasks

		#Optional function with parameters as in default_metric that returns whether to advance stage, and the accuracy / metric value
		self.metric = kwargs.get('metric', default_metric)
		
		#Optional list of accuracy functions to use for each task
		self.accuracies = kwargs.get('accuracy', [tasks[i].accuracy_function for i in range(len(tasks))])
		assert len(self.accuracies)==len(self.tasks)
		
		# Optional list of accuracy cuttoff values to use with each tasks
		self.thresholds = kwargs.get('thresholds', [.9 for i in range(len(tasks))])
		assert len(self.thresholds)==len(self.tasks)
		
		# How often to check metric?
		self.metric_epoch = kwargs.get('metric_epoch', 10)
		
		# Optional path to save out metric value and stage to
		self.output_file = kwargs.get('output_file', None)
		if self.output_file is not None:
			if path.dirname(self.output_file) != "" and not path.exists(path.dirname(self.output_file)):
				makedirs(path.dirname(self.output_file))
	
	def metric_test(self, input_data, correct_output, output_mask, test_output, epoch, losses, verbosity = False):
		advance, metric_value = self.metric(self.__dict__, input_data, correct_output, output_mask, test_output, epoch, losses, verbosity)
		self.metric_values.append([metric_value, self.stage])
		if advance:
			self.stage+=1
			if self.stage == len(self.tasks):
				self.stopTraining = True
				if self.output_file is not None:
					np.save(self.output_file, self.metric_values)
					if verbosity:
						print("Metric values saved in file: %s" % self.output_file)
			if verbosity:
				print("Stage " + str(self.stage))
			return True
		return False

	def get_generator_function(self):
		return self.tasks[self.stage].batch_generator()

	