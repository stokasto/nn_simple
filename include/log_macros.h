#ifndef LOG_MACROS_H_
#define LOG_MACROS_H_

#include <cstdio>
#include <iostream>
#include <vector>

// some useful log macros
// NOTE that the log level can be set via the DEBUG define
// logging can be completely disabled by setting the #if 1 
// to 0

#if 1
#define DEBUG 1
#define LOG(LVL,XX) \
  if ( LVL <= DEBUG ) \
  { \
    std::cerr << "LOG " << LVL << ": " XX << std::endl; \
  }
#define LOGF(LVL,FORM,...) \
  if ( LVL <= DEBUG ) \
  { \
    fprint(stderr,"LOG %d: ", LVL); \
    fprintf(stderr,FORM,__VA_ARGS__); \
    fprint(stderr,"\n"); \
  }
#else
#define LOG(LVL,XX)
#define LOGF(LVL,FORM,...)
#endif

#define ERROR(...) fprintf(stderr, __VA_ARGS__);


#endif /* LOG_MACROS_H_ */
