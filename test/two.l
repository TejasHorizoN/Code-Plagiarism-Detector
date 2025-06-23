 
%{ 
   #include<stdio.h> 
%} 
%% 
^[A-Za-z_][a-zA-Z0-9_]*$ { printf("valid : %s\n", yytext);} 
. { printf("invalid : %s", yytext);} 
%% 
int main() {    yylex();   return 0; } 
int yywrap() 
{  return 1; 
} 
