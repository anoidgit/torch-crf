local ffi = require("ffi")

ffi.cdef[[
void viterbiRoute(pfloat** trans, pfloat*** emit, pfloat* sos, pfloat* eos, int* seql, int ncondition, int*** rcache, pfloat** scache, int** route, pfloat* score);
void routeScore(int** route, pfloat** trans, pfloat*** emit, pfloat* sos, pfloat* eos, int* seql, int ncondition, pfloat* rs);
]]

local cAPI = ffi.load('libtcrf')

local CRFM, parent = torch.class('nn.CRFM', 'nn.Module')

function CRFM:__init(nstatus, weight)
	self:reset(weight or torch.randn(nstatus + 2, nstatus))
	self.nstatus = nstatus
	self.stdZero = torch.zeros(nstatus)
end

function CRFM:updateOutput(input)
	local function C2Table(cdata, fdim, sdim)
		local rs = {}
		for i = 0, fdim - 1 do
			local curd = {}
			for j = 0, sdim - 1 do
				table.insert(curd, cdata[i][j])
			end
			table.insert(rs, curd)
		end
		return rs
	end
	self:prepare(input)
	cAPI.viterbiRoute(self.cweight, self.cinput, self.trans[self.nstatus], self.trans[self.nstatus + 1], self.cseql, self.nstatus, self.rcache, self.scache, self.coutput, self.score)
	self.output = torch.IntTensor(C2Table(self.coutput, self.bsize, self.seql)):t():typeAs(input)
	return self.output
end

function CRFM:updateGradInput(input, gradOutput)
	if not self.gradInput:isSize(input) then
		self.gradInput:resizeAs(input):zero()
	else
		self.gradInput:zero()
	end
	for i = 0, self.bsize -1 do
		local _loss = self.loss[i + 1]
		local _nloss = -_loss
		for j = 0, self.cseql[i] - 1 do
			if self.cgold[i][j] ~= self.coutput[i][j] then
				self.gradInput[j + 1][i + 1][self.coutput[i][j]] = _loss
				self.gradInput[j + 1][i + 1][self.cgold[i][j]] = _nloss
			end
		end
	end
	return self.gradInput
end

function CRFM:accGradParameters(input, gradOutput, scale)
	scale = scale or 1
	for i = 0, self.bsize -1 do
		local _loss = self.loss[i + 1] * scale
		if self.cgold[i][0] ~= self.coutput[i][0] then
			self.gradWeight[self.nstatus][self.coutput[i][0]] = self.gradWeight[self.nstatus][self.coutput[i][0]] + _loss
			self.gradWeight[self.nstatus][self.cgold[i][0]] = self.gradWeight[self.nstatus][self.cgold[i][0]] - _loss
		end
		for j = 1, self.cseql[i] - 2 do
			if self.cgold[i][j] ~= self.coutput[i][j] then
				self.gradWeight[self.coutput[i][j - 1]][self.coutput[i][j]] = self.gradWeight[self.coutput[i][j - 1]][self.coutput[i][j]] + _loss
				self.gradWeight[self.cgold[i][j - 1]][self.cgold[i][j]] = self.gradWeight[self.cgold[i][j - 1]][self.cgold[i][j]] - _loss
			end
		end
		if self.cgold[i][self.cseql[i] - 1] ~= self.coutput[i][self.cseql[i] - 1] then
			self.gradWeight[self.coutput[i][self.cseql[i] - 2]][self.coutput[i][self.cseql[i] - 1]] = self.gradWeight[self.coutput[i][self.cseql[i] - 2]][self.coutput[i][self.cseql[i] - 1]] + _loss
			self.gradWeight[self.cgold[i][self.cseql[i] - 2]][self.cgold[i][self.cseql[i] - 1]] = self.gradWeight[self.cgold[i][self.cseql[i] - 2]][self.cgold[i][self.cseql[i] - 1]] - _loss
			self.gradWeight[self.nstatus + 1][self.coutput[i][self.cseql[i] - 1]] = self.gradWeight[self.nstatus + 1][self.coutput[i][self.cseql[i] - 1]] + _loss
			self.gradWeight[self.nstatus + 1][self.cgold[i][self.cseql[i] - 1]] = self.gradWeight[self.nstatus + 1][self.cgold[i][self.cseql[i] - 1]] - _loss
		end
	end
end

function CRFM:prepare(input)
	local isize = input:size()
	local seql = isize[1]
	local bsize = isize[2]
	self.cinput = ffi.new(string.format("float[%d][%d][%d]", bsize, seql, self.nstatus), input:transpose(1, 2):totable)
	self.cweight = ffi.new(string.format("float[%d][%d]", self.nstatus + 2, self.nstatus), self.weight:totable())
	self:getSeqlen(input, bsize, seql)
	if (seql ~= self.seql) or (bsize ~= self.bsize) then
		self.rcache = ffi.new(string.format("int[%d][%d][%d]", bsize, seql - 1, self.nstatus))
		self.scache = ffi.new(string.format("float[%d][%d]", bsize, seql))
		self.coutput = ffi.new(string.format("int[%d][%d]", bsize, seql))
		self.seql = seql
		if bsize ~= self.bsize then
			self.cscore = ffi.new(string.format("float[%d]", bsize))
			self.bsize = bsize
		end
	end
end

function CRFM:computeGold(gold)
	self.cgold = ffi.new(string.format("int[%d][%d]", self.bsize, self.seql), gold:t():totable())
	self.cgscore = ffi.new(string.format("float[%d]", self.bsize))
	cAPI.routeScore(self.cgold, self.cweight, self.cinput, self.trans[self.nstatus], self.trans[self.nstatus + 1], self.cseql, self.nstatus, self.cgscore)
	return self.cgscore
end

function CRFM:getLoss(gold, avg)
	self:computeGold(gold)
	self.loss = {}
	local loss = 0
	for i = 0, self.bsize - 1 do
		local _loss = self.cscore[i] - self.cgscore[i]
		table.insert(self.loss, _loss)
		loss = loss + _loss
	end
	if avg then
		local wds = 0
		for i = 0, self.bsize - 1 do
			wds = wds + self.cseql[i]
		end
		loss = loss / wds
	end
	return loss
end

function CRFM:getSeqlen(seqd, bsize, seql)
	local seqlen = {}
	for i = 1, bsize do
		for j = seql, 1, -1 do
			if not seqd[j][i]:equal(self.stdZero) then
				table.insert(seqlen, j)
				break
			end
		end
	end
	self.cseql = ffi.new(string.format("int[%d]", bsize), seqlen)
end

function CRFM:reset(weight)
	-- compute logSoftMax of weight
	local w = weight:reshape(weight:size(1) * weight:size(2))
	w:csub(w:max()):exp()
	w:div(w:sum()):log()
	-- asign logsoftmax probability to self.weight
	self.weight = w:reshape(weight:size())
	self.gradWeight:resizeAs(self.weight):zero()
	self:clearState()
end

function CRFM:clearState()
	self.loss = nil
	self.seql = 0
	self.bsize = 0
	self.cinput = nil
	self.cweight = nil
	self.cseql = nil
	self.rcache = nil
	self.scache = nil
	self.coutput = nil
	self.cscore = nil
	self.cgold = nil
	self.cgscore = nil
	return parent.clearState()
end
