#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>

#include "IREEGemm/Compile.h"

typedef struct compiler_state_t {
  iree_compiler_session_t* session;
  iree_compiler_source_t* source;
  iree_compiler_output_t* output;
  iree_compiler_invocation_t* inv;
} compiler_state_t;

int handleError(iree_compiler_error_t* error, compiler_state_t* s) {
  if (!error) return 0;
  const char* msg = ireeCompilerErrorGetMessage(error);
  fprintf(stderr, "Error from compiler API:\n%s\n", msg);
  ireeCompilerErrorDestroy(error);
  ireeGemmCompilerCleanup(s);
  return 1;
}

compiler_state_t* ireeGemmCompilerInitialize(int argc, char** argv) {
  if (!ireeCompilerLoadLibrary(
          "/home/sujasper/rocm-gemm-benchmark/src/ireekernelsbuild/build/"
          "iree/lib/libIREECompiler.so")) {
    fprintf(stderr, "** Failed to initialize IREE Compiler **\n");
    exit(1);
  }

  ireeCompilerGlobalInitialize();

  compiler_state_t* s = (compiler_state_t*)malloc(sizeof(compiler_state_t));
  s->inv = NULL;
  s->output = NULL;
  s->source = NULL;
  s->session = ireeCompilerSessionCreate();

  ireeCompilerSessionSetFlags(s->session, argc, (const char* const*)argv);

  return s;
}

int ireeGemmCompilerCompile(compiler_state_t* s, const char* inputFilePath,
                            const char* outputFilePath) {
  s->source = NULL;
  int error = handleError(
      ireeCompilerSourceOpenFile(s->session, inputFilePath, &s->source), s);
  if (error) return error;

  s->inv = ireeCompilerInvocationCreate(s->session);
  ireeCompilerInvocationEnableConsoleDiagnostics(s->inv);
  if (!ireeCompilerInvocationParseSource(s->inv, s->source)) {
    fprintf(stderr, "Error parsing input source into invocation\n");
    ireeGemmCompilerCleanup(s);
    return 1;
  }

  if (!ireeCompilerInvocationPipeline(s->inv, IREE_COMPILER_PIPELINE_STD)) {
    fprintf(stderr, "Error running compiler invocation\n");
    ireeGemmCompilerCleanup(s);
    return 1;
  }

  s->output = NULL;
  error =
      handleError(ireeCompilerOutputOpenFile(outputFilePath, &s->output), s);
  if (error) return error;

  error =
      handleError(ireeCompilerInvocationOutputVMBytecode(s->inv, s->output), s);
  if (error) return error;

  ireeCompilerOutputKeep(s->output);

  ireeCompilerSourceDestroy(s->source);
  s->source = NULL;
  ireeCompilerOutputDestroy(s->output);
  s->output = NULL;

  return 0;
}

void ireeGemmCompilerCleanup(compiler_state_t* s) {
  if (s->inv) ireeCompilerInvocationDestroy(s->inv);
  if (s->output) ireeCompilerOutputDestroy(s->output);
  if (s->source) ireeCompilerSourceDestroy(s->source);
}

void ireeGemmCompilerShutdown(compiler_state_t* s) {
  if (s->session) ireeCompilerSessionDestroy(s->session);
  ireeCompilerGlobalShutdown();
}