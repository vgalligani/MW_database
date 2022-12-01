
% FORMAT   paths = SwitchUser(nflag)
%
% It automatically defines the user, but the user has
% to manually set her/his working directories
%
% OUT   paths  Structure with paths to files/folders
%	       depending on user
% IN    nflag  Flagname specifying the output folder
%              depending on the type of simulation
%
% 20.05.2020 Vasileios Barlakas
%
function paths = SwitchUser(nflag)

%- Define user and paths
%- Location of ARTS controlfiles
paths.includes = '/home/victoria.galligani/Work/Software/ARTS/ARTS_032021/ARTS_032021/controlfiles/general';
%- Main output path
paths.mfolder  = [pwd,'/','Output','/',char(nflag)];
%- Creates main output folder 
if ~exist(paths.mfolder, 'dir')
	mkdir(paths.mfolder);
end
