local ffi = require("ffi")

ffi.cdef[[
void viterbiRoute(pfloat** trans, pfloat*** emit, pfloat* sos, pfloat* eos, int* seql, int ncondition, int*** rcache, pfloat** scache, int** route, pfloat* score);
]]

local cAPI = ffi.load('libtcrf')

local CRFM, parent = torch.class('nn.CRFM', 'nn.Module')

function CRFM:__init(nstatus, weight)
	self:reset(weight or torch.randn(nstatus + 2, nstatus))
	self.nstatus = nstatus
	self.stdZero = torch.zeros(nstatus)
end

local function C2Tensor(cdata, fdim, sdim)
	local rs = {}
	for i = 0, fdim - 1 do
		local curd = {}
		for j = 0, sdim - 1 do
			table.insert(curd, cdata[i][j])
		end
		table.insert(rs, curd)
	end
	return torch.FloatTensor(rs)
end

function CRFM:updateOutput(input)
	self:prepare(input)
	cAPI.viterbiRoute(self.cweight, self.cinput, self.trans[self.nstatus], self.trans[self.nstatus + 1], self.cseql, self.nstatus, self.rcache, self.scache, self.coutput, self.score)
	self.output = C2Tensor(self.coutput, self.bsize, self.seql):t():typeAs(input)
	return self.output
end

function CRFM:prepare(input)
	local isize = input:size()
	local seql = isize[1]
	local bsize = isize[2]
	self.cinput = ffi.new(string.format("float[%d][%d][%d]", bsize, seql, self.nstatus), input:transpose(1, 2):totable)
	self.cweight = ffi.new(string.format("float[%d][%d]", self.nstatus + 2, self.nstatus), self.weight:totable())
	self:getSeqlen(input, bsize, seql)
	if (seql ~= self.seql) or (bsize ~= self.bsize) then
		self.rcache = ffi.new(string.format("int[%d][%d][%d]", bsize, seql, self.nstatus))
		self.scache = ffi.new(string.format("float[%d][%d]", bsize, seql))
		self.coutput = ffi.new(string.format("int[%d][%d]", bsize, seql))
		self.seql = seql
		if bsize ~= self.bsize then
			self.cscore = ffi.new(string.format("float[%d]", bsize))
			self.bsize = bsize
		end
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
	self.cseql = ffi.new(string.format("int[%d]", bsize), seqlen)
end

function CRFM:reset(weight)
	self.weight = nn.LogSoftMax():updateOutput(weight)
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
	return parent.clearState()
end
