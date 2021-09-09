close all;
clear all;
clc;

bid_root = 'BID';
clive_root = 'ChallengeDB_release';
koniq_root = 'koniq-10k';
spaq_root = 'SPAQ';

for session = 1:1
    
    filename = fullfile(clive_root,'splits2',num2str(session),'clive_train.txt');
    fid = fopen(filename);
    clive_data=textscan(fid,'%s%s%f%f%f%d');
    fclose(fid);
    
    filename = fullfile(bid_root,'splits2',num2str(session),'bid_train.txt');
    fid = fopen(filename);
    bid_data=textscan(fid,'%s%s%f%f%f%d');
    fclose(fid);
    
    filename = fullfile(koniq_root,'splits2',num2str(session),'koniq10k_train.txt');
    fid = fopen(filename);
    koniq_data=textscan(fid,'%s%s%f%f%f%d');
    fclose(fid);
    
    filename = fullfile(spaq_root,'splits2',num2str(session),'spaq_train.txt');
    fid = fopen(filename);
    spaq_data=textscan(fid,'%s%s%f%f%f%d');
    fclose(fid);
    
    fid = fopen(fullfile('./splits2',num2str(session),'train.txt'),'w');
    
%     %clive
    for i = 1:length(clive_data{1,1})
        path1 = clive_data(1);
        path2 = clive_data(2);
        y = clive_data(3);
        std1 = clive_data(4);
        std2 = clive_data(5);
        path1 = path1{1,1};
        path2 = path2{1,1};
        y = y{1,1};
        yb = clive_data(6);
        yb = yb{1,1};
        std1 = std1{1,1};
        std2 = std2{1,1};
        path1 = fullfile(clive_root,path1{i,1});
        path2 = fullfile(clive_root,path2{i,1});
        path1 = strrep(path1, '\', '/');
        path2 = strrep(path2, '\', '/');
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1, path2,y(i,1), std1(i,1), std2(i,1),yb(i,1));
    end
   % bid
    for i = 1:length(bid_data{1,1})
        path1 = bid_data(1);
        path2 = bid_data(2);
        y = bid_data(3);
        std1 = bid_data(4);
        std2 = bid_data(5);
        path1 = path1{1,1};
        path2 = path2{1,1};
        y = y{1,1};
        yb = bid_data(6);
        yb = yb{1,1};
        std1 = std1{1,1};
        std2 = std2{1,1};
        path1 = fullfile(bid_root,path1{i,1});
        path2 = fullfile(bid_root,path2{i,1});
        path1 = strrep(path1, '\', '/');
        path2 = strrep(path2, '\', '/');
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1, path2,y(i,1), std1(i,1), std2(i,1),yb(i,1));
    end
    %koniq10k
    for i = 1:length(koniq_data{1,1})
        path1 = koniq_data(1);
        path2 = koniq_data(2);
        y = koniq_data(3);
        std1 = koniq_data(4);
        std2 = koniq_data(5);
        path1 = path1{1,1};
        path2 = path2{1,1};
        y = y{1,1};
        yb = koniq_data(6);
        yb = yb{1,1};
        std1 = std1{1,1};
        std2 = std2{1,1};
        path1 = fullfile(koniq_root,path1{i,1});
        path2 = fullfile(koniq_root,path2{i,1});
        path1 = strrep(path1, '\', '/');
        path2 = strrep(path2, '\', '/');
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1, path2,y(i,1), std1(i,1), std2(i,1),yb(i,1));
    end
    %spaq
    for i = 1:length(spaq_data{1,1})
        path1 = spaq_data(1);
        path2 = spaq_data(2);
        y = spaq_data(3);
        std1 = spaq_data(4);
        std2 = spaq_data(5);
        path1 = path1{1,1};
        path2 = path2{1,1};
        y = y{1,1};
        yb = spaq_data(6);
        yb = yb{1,1};
        std1 = std1{1,1};
        std2 = std2{1,1};
        path1 = fullfile(spaq_root,path1{i,1});
        path2 = fullfile(spaq_root,path2{i,1});
        path1 = strrep(path1, '\', '/');
        path2 = strrep(path2, '\', '/');
        fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%d\r',path1, path2,y(i,1), std1(i,1), std2(i,1),yb(i,1));
    end
    fclose(fid);
end

split = 1;

