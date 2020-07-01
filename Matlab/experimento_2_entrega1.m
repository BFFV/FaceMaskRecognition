%% 1] Extracci�n de caractar�sticas.
clear
caract_train = [];
caract_test = [];
caract_valid = [];
pmax = 0;

archivo = 'Fotos/FaceMask166/';
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
% Definition of LBP featuresf
%se hace invariante a la rotacion
opLBP.vdiv        = 6;           % 3 vertical divition
opLBP.hdiv        = 8;           % 3 horizontal   divition
opLBP.samples     = 8;           % number of neighbor samples
opLBP.mappingtype = 'u2';        %uniforme con 59 elementos
%opLBP.weight      = 0;
opLBP.type        = 2;           % intensity feature

% % Definition of Haralick Features (28 caracter�sticas)
% opHar.dharalick   = 3;           % distances 3 pixels
% opHar.type        = 2;           % intensity feature
% 
% % Definition of Gabor Features
% opGab.Lgabor      = 8;           % number of rotations
% opGab.Sgabor      = 8;           % number of dilations (scale)
% opGab.fhgabor     = 2;           % highest frequency of interest
% opGab.flgabor     = 0.1;         % lowest frequency of interest
% opGab.Mgabor      = 21;          % mask size
% opGab.show        = 0;
% opGab.type        = 2;           % intensity feature
% 
% %Par�metros para HOG
% 
% options.nj    = 10;             % 10 x 10
% options.ni    = 10;             % histograms
% options.B     = 9;              % 9 bins 
% options.show  = 0;              % show results
% 
% 

%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Feature Extraction
%%%%%%%%%%%%%%%%%%%%%%%%

bf(1).name = 'lbp';      bf(1).options = opLBP;
% bf(2).name = 'haralick'; bf(2).options = opHar;
% bf(3).name = 'gabor';    bf(3).options = opGab;
opfx.b = bf;
opfx.colstr = 'g';

% All features from all images are extracted:
[f,fn,S] = Bfx_files_2(f1,opfx); % f: feature matrix, fn: feature names, S: image file names
% 
% %Faltan las caractaristicas de HOG
% caracteristicas_HOG = [];
% 
% for z = 1:f1.imgmax
%     I = Bio_loadimg_2(f1,z);
%     %caracteristicas de HOG
%     X = Bfx_hog(I,options);
% 
%     caracteristicas_HOG = [caracteristicas_HOG ; X ];
% 
% end        
% 
% %Unir HOG con LBP
% f = [f caracteristicas_HOG];

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
%probar clasificadores
% 2] Eliminar columnas altamente correlacionadas

%training
s1 = Bfs_clean(caract_train,0); %indices
caract_train_1 = caract_train(:,s1);  % selected features

%validation
caract_valid_1=caract_valid(:,s1);
%test
caract_test_1=caract_test(:,s1);
% 3] Normalizar 
%train
[caract_train_2,a,b] = Bft_norm(caract_train_1,0);

%valid
N = size(caract_valid_1,1);
caract_valid_2 = caract_valid_1.*(ones(N,1)*a) + ones(N,1)*b;
%test
N = size(caract_test_1,1);
caract_test_2 = caract_test_1.*(ones(N,1)*a) + ones(N,1)*b;

% Selecci�n de caracter�sticas

% %Partimos con SFS y escogemos las 50 primeras caracter�sticas
% 
% op.m = 20;                     % 50 escogidas
% op.b.name = 'fisher';          % SFS with Fisher
% op.show = 0;
% s = Bfs_sfs(caract_train_2,d1,op);       % index of selected features
% 
% caract_train_6 = caract_train_2(:,s);    % selected features
% zzz = 3;
% %Datos de testing
% caract_test_6 = caract_test_2(:,s);    % selected features
% %Datos de validacion
% caract_valid_6 = caract_valid_2(:,s);    % selected features
% 
% 
% % Luego seguimos con PCA y se escogen las 10 mejores
% 
% [caract_train_4,lambda,A,Xs,mx] = Bft_pca(caract_train_3,100);
% 
% N = size(caract_test_3,1);
% caract_test_4 = (caract_test_3 - ones(N,1)*mx)*A(:,1:100);
% 
% N = size(caract_valid_3,1);
% caract_valid_4 = (caract_valid_3 - ones(N,1)*mx)*A(:,1:100);
% 
% 
% %Por �ltimo, se juntan las carater�sticas seleccionadas por PCA y SFS y se pasan por SFS  nuevamente
% 
% caract_train_5 = [caract_train_3 caract_train_4];
% 
% op.m = 30;                     % 15 escogidas
% op.b.name = 'fisher';          % SFS with Fisher
% op.show = 0;
% s = Bfs_sfs(caract_train_5,d1,op);       % index of selected features
% 
% caract_train_6 = caract_train_5(:,s);    % selected features

% %Datos de testing
% caract_test_5 = [caract_test_3 caract_test_4];
% 
% caract_test_6 = caract_test_5(:,s);    % selected features
% 
% %Datos valid
% caract_valid_5 = [caract_valid_3 caract_valid_4];
% 
% caract_valid_6 = caract_valid_5(:,s);    % selected features

for p = 1:2

    if p == 1
        nombre = 'KNN';
        opcl.k = 6;
        ds = Bcl_knn(caract_train_2,d1 ,caract_valid_2, opcl);   

    elseif p == 2
        nombre = 'SVM';
        op.kernel = ['-t ' 1]; %se utiliza una transformacion polinomial
        ds = Bcl_libsvm(caract_train_2,d1,caract_valid_2,op);   % SVM  

    elseif p == 3
        nombre = 'QDA';
        opd.p = [];
        opd = Bcl_qda(caract_train_6,d1,opd);
        ds = Bcl_qda(caract_valid_6,opd);                    
    end

    per = Bev_performance(ds,dv);
    fprintf('Performance claisificador %s es = %f',nombre,per);

    if per > pmax
        clasificador = nombre;
        pmax = per;
        ddef = ds;
    end
end

        

%%


% opLBP.vdiv        = div_vertical;           % 3 vertical divition
% opLBP.hdiv        = div_horizontal;           % 3 horizontal   divition
% opLBP.samples     = 8;           % number of neighbor samples
% opLBP.mappingtype = 'u2';        %uniforme con 59 elementos
% %opLBP.weight      = 0;
% opLBP.type        = 2;           % intensity feature

%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Feature Extraction
%%%%%%%%%%%%%%%%%%%%%%%%


% bf(1).name = 'lbp';      bf(1).options = opLBP;
% 
% opfx.b = bf;
% opfx.colstr = 'g';
% 
% % All features from all images are extracted:
% [f,fn,S] = Bfx_files_2(f1,opfx); % f: feature matrix, fn: feature names, S: image file names
% caract_train =[];
% caract_test =[];
% caract_valid =[];
% for k = 1:6
%     d = 0;
%     for c = 1:16
%         if k == 1 || k == 2 || k == 3
% 
%             caract_train = [caract_train; f(k + d, :)];
%             d = d + 6;
%         elseif k == 4
%             caract_valid = [caract_valid; f(k + d, :)];
%             d = d + 6;   
%         else
%             caract_test = [caract_test; f(k + d, :)];
%             d = d + 6;                    
%         end
%     end
% end
% d1 = [];
% for o = 1:3
%     d1 = [d1 ; double(Bds_labels(1*ones(16,1)))]; % train
% end
% dv = double(Bds_labels(1*ones(16,1))); % 2 validation   

dt = [];
for o = 1:2
    dt = [dt ; double(Bds_labels(1*ones(16,1)))]; % test
end

%clasificadores

nombre = 'SVM';
op.kernel = ['-t ' 1]; %se utiliza una transformacion polinomial
ds = Bcl_libsvm(caract_train_2,d1,caract_test_2,op);   % SVM  


per = Bev_performance(ds,dt);
fprintf('Performance claisificador %s, hdiv: %u , vdiv: %u es = %f',nombre, j, i, per);

%Matriz de confusion
[T,p] = Bev_confusion(dt,ds);