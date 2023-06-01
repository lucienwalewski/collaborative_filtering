from submission import submit

class SubmissionPipeline:
    def __init__(self, preds):
        self.preds = preds

    def execute(self):
        submit(self.preds)