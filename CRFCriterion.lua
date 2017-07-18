local CRFCriterion, parent = torch.class('nn.CRFCriterion', 'nn.Criterion')

function CRFCriterion:__init(CRFM, sizeAverage)
	self.network = CRFM
	self.sizeAverage = sizeAverage
end

function CRFCriterion:updateOutput(input, target)
	self.output = self.network:getLoss(target)
	return self.output
end

-- loss and gradient is computed by crfm itself
function CRFCriterion:updateGradInput(input, target)
	self.gradInput = input.new():resizeAs(input):zero()
	return self.gradInput
end
