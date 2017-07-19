local ffi = require("ffi")

ffi.cdef[[
void viterbiRoute(float** trans, float*** emit, float* sos, float* eos, int bsize, int* seql, int ncondition, int*** rcache, float** scache, int** route, float* score);
void routeScore(int** route, float** trans, float*** emit, float* sos, float* eos, int bsize, int* seql, int ncondition, float* rs);
void calcGrad(int** gold, int** pred, int bsize, int* seql, float* losses, float** grad, float* gsos, float* geos);
float getLoss(float* gscore, float* pscore, int bsize, int* seql, int avg, float* losses);
void fillMat(float** mat, int fdim, int sdim);
]]

local cAPI = ffi.load('libtcrf')

local CRFM, parent = torch.class('nn.CRFM', 'nn.Container')

function CRFM:__init(nstatus, weight)
	parent.__init(self)
	self.network = nn.LogSoftMax()
	self:add(self.network)
	self.nstatus = nstatus
	self:reset(weight)
	self.stdZero = torch.zeros(nstatus)
end

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

local function getFloatMat(fdim, sdim, matin, ftyp, styp)
	local rs = ffi.new(ftyp or "float* ["..tostring(fdim).."]")
	local styp = styp or "float ["..tostring(sdim).."]"
	if matin then
		for i, vec in ipairs(matin) do
			rs[i - 1] = ffi.new(styp, vec)
		end
	else
		for i = 0, fdim - 1 do
			rs[i] = ffi.new(styp)
		end
	end
	return rs
end

local function getIntMat(fdim, sdim, matin, ftyp, styp)
	local rs = ffi.new(ftyp or "int* ["..tostring(fdim).."]")
	local styp = styp or "int ["..tostring(sdim).."]"
	if matin then
		for i, vec in ipairs(matin) do
			rs[i - 1] = ffi.new(styp, vec)
		end
	else
		for i = 0, fdim - 1 do
			rs[i] = ffi.new(styp)
		end
	end
	return rs
end

local function getFloatTensor(fdim, sdim, tdim, tin)
	local rs = ffi.new("float** ["..tostring(fdim).."]")
	local ftyp = "float* ["..tostring(sdim).."]"
	local styp = "float ["..tostring(tdim).."]"
	if tin then
		for i, mat in ipairs(tin) do
			rs[i - 1] = getFloatMat(sdim, tdim, mat, ftyp, styp)
		end
	else
		for i = 0, fdim - 1 do
			rs[i] = getFloatMat(sdim, tdim, nil, ftyp, styp)
		end
	end
	return rs
end

local function getIntTensor(fdim, sdim, tdim, tin)
	local rs = ffi.new("int** ["..tostring(fdim).."]")
	local ftyp = "int* ["..tostring(sdim).."]"
	local styp = "int ["..tostring(tdim).."]"
	if tin then
		for i, mat in ipairs(tin) do
			rs[i - 1] = getIntMat(sdim, tdim, mat, ftyp, styp)
		end
	else
		for i = 0, fdim - 1 do
			rs[i] = getIntMat(sdim, tdim, nil, ftyp, styp)
		end
	end
	return rs
end

function CRFM:updateOutput(input)
	self:prepare(input)
	cAPI.viterbiRoute(self.cweight, self.cinput, self.cweight[self.nstatus], self.cweight[self.nstatus + 1], self.bsize, self.cseql, self.nstatus, self.rcache, self.scache, self.coutput, self.score)
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
	if not self.cgrad then
		self.cgrad = getFloatMat(self.nstatus + 2, self.nstatus)
	else
		cAPI.fillMat(self.cgrad, self.nstatus + 2, self.nstatus)
	end
	cAPI.calcGrad(self.cgold, self.coutput, self.bsize, self.cseql, self.closs, self.cgrad, self.cgrad[self.nstatus], self.cgrad[self.nstatus + 1])
	self.gradWeight:add(scale or 1, self.network:updateGradInput(self.weight, torch.FloatTensor(C2Table(self.cgrad, self.nstatus + 2, self.nstatus)):typeAs(self.weight)))
end

function CRFM:prepare(input)
	local isize = input:size()
	local seql = isize[1]
	local bsize = isize[2]
	self.cinput = getFloatTensor(bsize, seql, self.nstatus, input:transpose(1, 2):totable())
	self.cweight = getFloatMat(self.nstatus + 2, self.nstatus, self.network:updateOutput(self.weight):totable())
	self:getSeqlen(input, bsize, seql)
	if (seql ~= self.seql) or (bsize ~= self.bsize) then
		self.rcache = getIntTensor(bsize, seql - 1, self.nstatus)
		self.scache = getFloatMat(bsize, seql)
		self.coutput = getIntMat(bsize, seql)
		self.seql = seql
		if bsize ~= self.bsize then
			self.cscore = ffi.new("float ["..tostring(bsize).."]")
			self.bsize = bsize
		end
	end
end

function CRFM:computeGold(gold)
	self.cgold = getIntMat(self.bsize, self.seql, gold:t():totable())
	self.cgscore = ffi.new("float ["..tostring(self.bsize).."]")
	cAPI.routeScore(self.cgold, self.cweight, self.cinput, self.trans[self.nstatus], self.trans[self.nstatus + 1], self.bsize, self.cseql, self.nstatus, self.cgscore)
	return self.cgscore
end

function CRFM:getLoss(gold, avg)
	self:computeGold(gold)
	self.closs = ffi.new("float ["..tostring(self.bsize).."]")
	if avg then
		return cAPI.getLoss(self.cgscore, self.cscore, self.bsize, self.cseql, 1, self.closs)
	else
		return cAPI.getLoss(self.cgscore, self.cscore, self.bsize, self.cseql, 0, self.closs)
	end
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
	self.cseql = ffi.new("int ["..tostring(bsize).."]", seqlen)
end

function CRFM:reset(weight)
	self.weight = weight or torch.randn(self.nstatus + 2, self.nstatus)
	self.gradWeight = self.weight.new():resizeAs(self.weight):zero()
	self:clearState()
end

function CRFM:clearState()
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
	self.cgrad = nil
	self.closs = nil
	return parent.clearState(self)
end
