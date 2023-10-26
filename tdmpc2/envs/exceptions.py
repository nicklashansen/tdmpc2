
class UnknownTaskError(Exception):
	def __init__(self, task):
		super().__init__(f'Unknown task: {task}')
