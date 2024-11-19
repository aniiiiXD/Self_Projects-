function varargout = zstddecode(varargin)
%
% output = zstddecode(input)
%    or
% output = zstddecode(input,info)
%
% Decompressing an Zstd-compressed byte-stream to recover the original data
% This function depends on the ZMat toolbox (http://github.com/NeuroJSON/zmat)
%
% authors:Qianqian Fang (q.fang <at> neu.edu)
%
% input:
%      input: a string, int8/uint8 vector or numerical array to store Zstd-compressed data
%      info (optional): a struct produced by the zmat/zstdencode function during
%            compression; if not given, the inputs/outputs will be treated as a
%            1-D vector
%
% output:
%      output: the decompressed byte stream stored in a uint8 vector; if info is
%            given, output will restore the original data's type and dimensions
%
% examples:
%      [bytes, info]=zstdencode(eye(10));
%      orig=zstddecode(bytes,info);
%
% license:
%     BSD or GPL version 3, see LICENSE_{BSD,GPLv3}.txt files for details
%
% -- this function is part of JSONLab toolbox (http://iso2mesh.sf.net/cgi-bin/index.cgi?jsonlab)
%

if (nargin == 0)
    error('you must provide at least 1 input');
end
if (exist('zmat', 'file') == 2 || exist('zmat', 'file') == 3)
    if (nargin > 1)
        [varargout{1:nargout}] = zmat(varargin{1}, varargin{2:end});
    else
        [varargout{1:nargout}] = zmat(varargin{1}, 0, 'zstd', varargin{2:end});
    end
else
    error('you must install ZMat toolbox to use this feature: http://github.com/NeuroJSON/zmat');
end
