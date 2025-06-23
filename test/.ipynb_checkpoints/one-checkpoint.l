%{ 
   #include<stdio.h> 
   int lines=0, spaces = 0, chars = 0, tabs = 0; 
%} 
%% 
\n {lines++, chars++ ; } 
" " {spaces++, chars++ ; } 
\t {tabs++, chars++ ; } 
. {chars++;} 
%% 
int main() {    yylex(); 
  printf("lines= %d, tabs = %d, spaces = %d, chars %d", lines, tabs, spaces, chars);   return 0; } 
int yywrap() 
{  return 1; 
} 
  
