#include <iostream>
#include <cmath>
using namespace std;

long long ext_gcd(long long a,long long b,long long& x,long long& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    long long d = ext_gcd(b, a % b, x, y);
    x -= (a / b) * y;
    swap(x, y);
    return d;
}
 
int main(){
long long a,b,c,x,y,d;
cin>>a>>b>>c;
if (a < 1000000000 && b < 1000000000 && c < 1000000000 && a > 0 && b >0 && c > 0){
    d=ext_gcd(a,b,x,y);
    if(c%d==0){
        long long t = c/d*x,t2=b/d;
        if(t==0)cout<<0<<" "<<c/d*y;
        if(t >0)cout<<t+t2*(-(t/t2))<<" "<<c/d*y-a/d*(-(t/t2));
        if(t < 0)cout<<t+t2*(-((t-t2+1)/t2))<<" "<<c/d*y-a/d*((-((t-t2+1)/t2)));
    }
    else cout<<"Impossible";
    return 0;
    }
}
