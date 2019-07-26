class NLP():

	input_data = ""

	new_dict = []
	labels = []
	examples = []

	trained_data = []

	def __init__(self):
		return
		

	def set_labels(self, lab):

		self.labels = lab
		for label in lab:
			self.trained_data.append({label: 1})


	def set_examples(self, ex):
		global examples
		for example in ex:
			if len(example) >= 5:
				examples = ex
			else:
				print("Error: Must have at least 5 examples for each label.")

	def train(self):
		for i in range(0,len(examples)):
			for example in examples[i]:
				self.tokenize(example, i)

		
	def show_training_data(self):
		if len(self.trained_data) != 0:
			for item in self.trained_data:
				print(item)

	def tokenize(self, item, i):
		tokenized = item.split(" ")

		for word in tokenized:
			if word in self.trained_data[i]:
				count = self.trained_data[i][word] + 1
				self.trained_data[i][word] = count
			else:
				self.trained_data[i][word] = 1

	def classify(self, input):
		tokenize = input.lower().split(" ")
		scores = []


		for i in range(0, len(self.trained_data)):
			total = 0

			for word in tokenize:
				if word in self.trained_data[i]:
					total += 1

			scores.append(total)

		largest_index = 0
		largest_sum = 0

		for i in range(0, len(scores)):
			if largest_sum < scores[i]:
				largest_sum = scores[i]
				largest_index = i

		return self.labels[largest_index]
