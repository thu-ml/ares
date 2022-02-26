clc
clear all
L=10;
D=256;
mean_var=1;

mean_logits=zeros(L,D);
mean_logits(1,1)=1;
for k=2:L
    for j =1:k-1
        mean_logits(k,j)=-(1/(L-1)+dot(mean_logits(k,:),mean_logits(j,:)))/mean_logits(j,j);
    end
    mean_logits(k,k)=sqrt(abs(1-norm(mean_logits(k,:))^2));
end
mean_logits=mean_logits*mean_var;


save(['meanvar1_featuredim',num2str(D),'_class',num2str(L),'.mat'],'mean_logits')
