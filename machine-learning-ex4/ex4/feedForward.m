function J = feedForward(X,y,Theta1,Theta2,lambda)
    [m,n]=size(X)
    
    a1=[ones(m,1) X];
    z2=a1 * Theta1';
    a2=sigmoid(z2);
    size(a2);
    a2=[ones(size(a2,1),1) a2];
    size(a2);
    z3= a2*Theta2';
    a3=z3;
    k=size(a3,2);
    J=0;
    
    for i=1:k
        y_temp=y==i;
        J=J+((dot(-y_temp,log(sigmoid(a3(:,i)))))-dot((1-y_temp),log(1-sigmoid(a3(:,i)))));
    end    
    J=(sum(J))/m;
    J=J + lambda/(2*m)* (sum(sum(Theta1(:,2:size(Theta1,2)).^2)) +sum(sum(Theta2(:,2:size(Theta2,2)).^2)));
    
end

