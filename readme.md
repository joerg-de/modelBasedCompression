# Model Based Compression API

## API Reference

### PytorchModel
- PytorchModel(std::string path)
  Loads a pytorch model model from disk.
- std::set<std::string> listU8Chars() const
  Returns the supported character set extracted from the model's vocabulary.
- void loadBlacklist(std::string path, bool capitalize = true, bool force = false)
  Loads a blacklist file.
- void loadBlacklistLine(const std::string& line, std::string& errorText, bool force)
  Loads a single blacklist line.
- std::shared_ptr<PytorchState> generateState(std::shared_ptr<PytorchModel> self)
  Creates a new state instance bound to this model. Pass the model's shared pointer to maintain reference validity.
- std::unique_ptr<PytorchModel> clone() const
  Deep-copies the model configuration.

### PytorchState
- void resetHidden()
  Resets internal hidden states. (Note: Python bindings exposes resetDecState(); use this for C++).
- std::shared_ptr<PytorchState> cloneState() const
  Duplicates the current decoder/encoder state. (Python bindings exposes cloneDecState()).

#### Huffman Compression (#ifdef SUPPORT_HUFFMAN)
- void initWithText(const char* text, std::function<void(unsigned long long)> callBack, bool updateBlacklist = false)
  Encodes text without outputing the result. callBack receives processed byte count.
- std::tuple<std::string, unsigned int, unsigned int> decodeToBinary(const char* text, std::function<void(unsigned long long)> callBack, bool useEndMarker = false, double corrVactor = 1, unsigned int HuffMaxExtend = 32, double huffMin = 0.0078125)
  Encodes text to binary Huffman stream. Returns {encoded_string, leftover_bytes, maxDestr<maximum bytes changed when text is appended to the output>}.
  **Limitation:** Do not call subsequent Decode/Encode operations after invoking this with useEndMarker=true. The state becomes terminal. Omitting it may cause encoding continuation failures.
- std::tuple<std::string, int> decodeFromBinary(const char* data, std::function<void(unsigned long long)> callBack, bool useEndMarker = false, double corrVactor = 1, unsigned int HuffMaxExtend = 32, double huffMin = 0.0078125)
  Decodes binary stream back to text. Returns {decoded_string, leftover_bytes}.

#### Range Coding (#ifdef SUPPORT_RANGE_CODING)
- void initRangeCoding(unsigned int NormalBufferSize = 64)
  **Mandatory:** Must be called before any range coding operations. Initializes arithmetic coder buffers.
- std::tuple<std::string, int> decodeToBinaryRangeCoding(const char* text, std::function<void(unsigned long long)> callBack, bool useEndMarker = false, unsigned int MinUsefullBitsSize = 32, unsigned int NormalBufferSize = 64, double corrVactor = 1, bool ignoreUnkToken = true)
  Encodes text using range coding. Returns {encoded_string, leftover_bytes}.
  **Limitation:** MinUsefullBitsSize must be strictly greater than log2(NumToken). ignoreUnkToken skips vocabulary misses during encoding.
- std::tuple<std::string, int> decodeFromBinaryRangeCoding(const char* data, std::function<void(unsigned long long)> callBack, bool useEndMarker = false, unsigned int MinUsefullBitsSize = 32, unsigned int NormalBufferSize = 64, long long sizeToEncode = 0, double corrVactor = 1)
  Decodes range-coded binary. Returns {decoded_string, leftover_bytes}. sizeToEncode optionally constrains the decoding window.
- BigUInt getMin() / BigUInt getMax()
  Returns current interval bounds.

### PytorchModelCache
- std::shared_ptr<PytorchModel> getModelAt(std::string path)
  Thread-safe singleton loader. Returns partly cached instance if already loaded, otherwise loads and caches it.

## PyTorch Model Preparation & Usage
The compression pipeline requires a compiled TorchScript model paired with auxiliary configuration files. These files can either exist alongside the `.pt` file or be embedded directly into the TorchScript archive.

### Required File Formats
- model.wor: Comma-separated token list. Supports escaping (\\ for backslash, \, for comma).
- model.pro: One probability value per line, strictly matching the token count in model.wor. Must be valid doubles (NaN throws on load).
- model.tmp: One integer per line representing hidden state/carryover sizes. Values must be > 0. Alternatively the model can have a init_hidden function if no tmp file is supplyed this function is used.
- model.opt: Three integers on separate lines: freez, optimize_numerics, optimize_for_inference. Controls JIT freezing and optimization flags during initialization.

### Embedding Mechanism
If external files are missing, the loader falls back to embedded resources stored in the TorchScript model's ExtraFilesMap. Use these exact keys when embedding: wor.wor, pro.pro, tmp.tmp, opt.opt
External files always take precedence over embedded ones. To embed, use TorchScript's save API:
torch::jit::save(model, "model.pt", extra_files={{"wor.wor", word_content}, {"pro.pro", prob_content}, ...});

## Important Notes & Limitations
- Callback signature: std::function<void(unsigned long long)>. The argument represents cumulative bytes processed during encoding/decoding.
- useEndMarker: once set, the state cannot be reused for further encode/decode cycles without resetting.

## Todo
- Null in tokens does not work yet. The text just ends there.
