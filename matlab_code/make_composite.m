function out = make_composite(func,anat,thresh,ker,gridsize)
%func and anat must have the same number of slices
%function out = make_composite(func,anat,thresh,ker)

func = squeeze(func);

if (~exist('ker','var')),
   ker = 'bilinear';
end;
   
if (~exist('gridsize','var')),
   gridsize = -999;
end;

colortables

xdim = size(anat,1);ydim = size(anat,2);zdim = size(anat,3);
xdimf = size(func,1);ydimf = size(func,2);zdimf = size(func,3);

%resize functional map to be larger (same as the template)
clear func2;
for j = 1:zdim,
   func2(:,:,j) = imresize(func(:,:,j),[xdim ydim],ker);
end;
func = func2;

%scale functional to range from 129-256
if (minN(func)>=0),
   %positive only
   v = [];
   ind = find(func>thresh);if (length(ind)<2), v = 0; end; 
   func = (128+(func.*(func>thresh)-minN([func(func>thresh);v]))* ...
      127/maxN(func-minN([func(func>thresh);v]))).*(func>thresh);
   clear gray;
   h = hot(384);
   aa = [gray(256);h(129:end,:)];
   %aa = [gray(256);imresize(cold(60:end,:),[256 3])];
   
   %gr = gray(1000);
   %aa = [gray(256);gr(end:-1:(end-255),:)];
else,
   %positive and negative
   [Y I]=max(flatrow(abs(func)));
   Y = func(I);

   func(1)=abs(Y);
   func(end)=-abs(Y);
   
   func = (128+(func.*(abs(func)>thresh)-minN(func))* ...
      127/maxN(func-minN(func))).*(abs(func)>thresh);
   func(1) = 0;func(end)=0;

   clear gray;
   aa = [gray(256);coldhot];
   
%    func = (128+(func.*(func>thresh)-minN(func))* ...
%       127/maxN(func-minN(func))).*(func>thresh);
%    clear gray;
%    aa = [gray(256);coldhot];
   
end;

%scale anatomic to range from 1-128
anat = (anat-minN(anat))*127/maxN(anat-minN(anat));

%combine them
out = anat.*(func==0)+func;

%pick which slices to display...
%out=out(:,:,:);

% c = [0 0 0;0 0 1;0 1 0;0 1 1;1 0 0;1 0 1;1 1 0;1 1 1];
% ind = 1;
% for j = 257:32:512,
%     aa(j:j+31,:) = repmat(c(ind,:),32,1);
%     ind = ind+1;
% end;

a=create_2d(out,size(out,1),size(out,2),size(out,3),1,size(out,3),1,1,gridsize,0,256,0);

%colormap(aa(1:2:end,:))
subimage(round(a')+1,aa(1:2:end,:));axis off;
%set(gcf,'position',[22   506   518   429]);
%print(gcf,'-dtiff',['nocrave_' num2str(jj,'%.2d')]);
