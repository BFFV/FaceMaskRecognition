%% 1] Extracción de caractarísticas.
caract_train = [];
caract_test = [];
caract_valid = [];
pmax = 0;

archivo = '../Fotos/FaceMask166/';
f1.imgmax    = 96;

% Definition of images para clasificador
f1.path      = archivo;
f1.extension = 'jpg';
f1.prefix    = '*';
f1.gray      = 1;
f1.imgmin    = 1;


%probar distintas divisiones horizontale y verticales

caract_train = [];
caract_test  = [];
caract_valid = [];        
% Definition of LBP features
%se hace invariante a la rotacion
opLBP.vdiv        = 6;           % 3 vertical divition
opLBP.hdiv        = 8;           % 3 horizontal   divition
opLBP.samples     = 8;           % number of neighbor samples
opLBP.mappingtype = 'u2';        %uniforme con 59 elementos
%opLBP.weight      = 0;
opLBP.type        = 2;           % intensity feature


%Parámetros para HOG

options.nj    = 10;             % 10 x 10
options.ni    = 10;             % histograms
options.B     = 9;              % 9 bins 
options.show  = 1;              % show results



%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Feature Extraction
%%%%%%%%%%%%%%%%%%%%%%%%

bf(1).name = 'lbp';      bf(1).options = opLBP;

opfx.b = bf;
opfx.colstr = 'g';

% All features from all images are extracted:
[f,fn,S] = Bfx_files_2(f1,opfx); % f: feature matrix, fn: feature names, S: image file names

%Faltan las caractaristicas de HOG
caracteristicas_HOG = [];

for z = 1:f1.imgmax
    I = Bio_loadimg_2(f1,z);
    %caracteristicas de HOG
    X = Bfx_hog(I,options);

    caracteristicas_HOG = [caracteristicas_HOG ; X ];

end        

for k = 1:6
    d = 0;
    for c = 1:16
        if k == 1 || k == 2 || k == 3

            caract_train = [caract_train; f(k + d, :)];
            d = d + 6;
        elseif k == 4
            caract_valid = [caract_valid; f(k + d, :)];
            d = d + 6;   
        else
            caract_test = [caract_test; f(k + d, :)];
            d = d + 6;                    
        end
    end
end

d1 = [];
for o = 1:3
    d1 = [d1 ; double(Bds_labels(1*ones(16,1)))]; % train
end
dv = double(Bds_labels(1*ones(16,1))); % 2 validation     
%ver probar clasificadores
for p = 1:3

    if p == 1
        nombre = 'KNN';
        opcl.k = 6;
        ds = Bcl_knn(caract_train,d1 ,caract_valid, opcl);   

    elseif p == 2
        nombre = 'SVM';
        op.kernel = ['-t ' 1]; %se utiliza una transformacion polinomial
        ds = Bcl_libsvm(caract_train,d1,caract_valid,op);   % SVM  

    elseif p == 3
        nombre = 'QDA';
        opd.p = [];
        opd = Bcl_qda(caract_train,d1,opd);
        ds = Bcl_qda(caract_valid,opd);                    
    end

    per = Bev_performance(ds,dv);
    fprintf('Performance claisificador %s, hdiv: %u , vdiv: %u es = %f',nombre, j, i, per);

    if per > pmax
        clasificador = nombre;
        pmax = per;
    end
end

        

%%
opLBP.vdiv        = div_vertical;           % 3 vertical divition
opLBP.hdiv        = div_horizontal;           % 3 horizontal   divition
opLBP.samples     = 8;           % number of neighbor samples
opLBP.mappingtype = 'u2';        %uniforme con 59 elementos
%opLBP.weight      = 0;
opLBP.type        = 2;           % intensity feature

%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Feature Extraction
%%%%%%%%%%%%%%%%%%%%%%%%


bf(1).name = 'lbp';      bf(1).options = opLBP;

opfx.b = bf;
opfx.colstr = 'g';

% All features from all images are extracted:
[f,fn,S] = Bfx_files_2(f1,opfx); % f: feature matrix, fn: feature names, S: image file names
caract_train =[];
caract_test =[];
caract_valid =[];
for k = 1:6
    d = 0;
    for c = 1:16
        if k == 1 || k == 2 || k == 3

            caract_train = [caract_train; f(k + d, :)];
            d = d + 6;
        elseif k == 4
            caract_valid = [caract_valid; f(k + d, :)];
            d = d + 6;   
        else
            caract_test = [caract_test; f(k + d, :)];
            d = d + 6;                    
        end
    end
end
d1 = [];
for o = 1:3
    d1 = [d1 ; double(Bds_labels(1*ones(16,1)))]; % train
end
dv = double(Bds_labels(1*ones(16,1))); % 2 validation   

dt = [];
for o = 1:2
    dt = [dt ; double(Bds_labels(1*ones(16,1)))]; % test
end

%clasificadores

nombre = 'SVM';
op.kernel = ['-t ' 1]; %se utiliza una transformacion polinomial
ds = Bcl_libsvm(caract_train,d1,caract_test,op);   % SVM  


per = Bev_performance(ds,dt);
fprintf('Performance claisificador %s, hdiv: %u , vdiv: %u es = %f',nombre, j, i, per);

%Matriz de confusion
[T,p] = Bev_confusion(dt,ds);

