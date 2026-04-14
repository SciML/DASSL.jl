_process_verbose_param(v::SciMLLogging.AbstractVerbosityPreset) = DEVerbosity(v)
_process_verbose_param(v::Bool) = v ? DEVerbosity() : DEVerbosity(SciMLLogging.None())
_process_verbose_param(v::DEVerbosity) = v
