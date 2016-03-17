//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include <Basics.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include "UCIParser.h"

#include <string>
#include <windows.h>
#include <File.h>
#include "Indexer.h"
#include "TextParser.h"


namespace Microsoft { namespace MSR { namespace CNTK {

// A thin wrapper around CNTK text format deserialzier (parser)
template <class ElemType>
class CNTKTextFormatReaderTestRunner
{
    TextParser<ElemType> m_parser;

    vector<typename TextParser<ElemType>::SequenceBuffer> m_sequences;
public:
    CNTKTextFormatReaderTestRunner(const std::wstring& filename, 
        const vector<StreamDescriptor>& streams, size_t chunkSize, unsigned int numChunks) :
        m_parser(filename, streams)
    {
        m_parser.SetMaxAllowedErrors(0);
        m_parser.SetChunkSize(chunkSize);
        m_parser.SetChunkCacheSize(numChunks);
        m_parser.Initialize();
    }
    
    // Description of streams that this data deserializer provides.
    const std::vector<SequenceDescriptor>& GetTimeline()
    {
        return m_parser.m_index->m_timeline;
    }

    void LoadSequence(const SequenceDescriptor& descriptor, vector<vector<ElemType>>& values)
    {
        m_sequences.push_back(move(m_parser.LoadSequence(false, descriptor)));
        auto const& sequence = m_sequences.back();
        for (int i = 0; i < sequence.size(); ++i)
        {
            const auto& data = sequence[i]->m_buffer;
            copy(data.begin(), data.end(), back_inserter(values[i]));
        }       
    }

    // Retrieves a chunk of data.
    ChunkPtr GetChunk(size_t chunkId) { return m_parser.GetChunk(chunkId);  }
};

}}}

namespace cntk = Microsoft::MSR::CNTK;

namespace fs = boost::filesystem;

template<typename Func>
size_t benchmark(Func func)
{
    long start = GetTickCount();
    func();
    return GetTickCount() - start;
}

void compareIndexingPerf(wstring filename)
{
    
    size_t count1 = 0, count2 = 0;

    cout << "Building an input data index." << endl;
    {
        UCIParser<float, int> uciParser(char(0), char(0));
        size_t bufSize = 256 * 1024; // same buffer size as in CNTKTextFormatReader
        // uci parser only counts the number of lines, so using the same file for both
        uciParser.ParseInit(filename.c_str(), 1, 784, 0, 1, bufSize);
        uciParser.SetParseMode(ParseMode::ParseLineCount);
        cout << "UCIFastReader : "
            << benchmark([&count1, &uciParser]() {count1 = uciParser.Parse(size_t(-1), NULL, NULL); })
            << " ms" << endl;
    }

    {
        cntk::Indexer textIndexer(fopenOrDie(filename, L"rbS"), false, SIZE_MAX);
        cntk::IndexPtr index;
        cout << "CNTKTextFormatReader: "
            << benchmark([&count2, &textIndexer, &index]()
            {
                index = textIndexer.Build();
                count2 = index->m_timeline.size();
            })
            << "ms" << endl;
    }

    if (count1 != count2)
    {
        RuntimeError("Output of the indexing operations do not match");
    }
}

void compareParsingPerfSinglePass(wstring uciFilename, wstring textFilename)
{
    cout << "Parsing input data (single pass). " << endl;
    {
        UCIParser<float, int> uciParser(char(0), char(0));
        size_t bufSize = 256 * 1024; // same buffer size as in CNTKTextFormatReader
        // uci parser only counts the number of lines, so using the same file for both
        uciParser.ParseInit(uciFilename.c_str(), 1, 784, 0, 1, bufSize);
        
        vector<float> values;
        values.reserve(784 * 10000);
        vector<int> labels;
        labels.reserve(10000);

        cout << "UCIFastReader : "
            << benchmark([&uciParser, &values, &labels]()
            {
                int records = 0;
                do
                {
                    int recordsRead = uciParser.Parse(10000, &values, &labels);
                    if (recordsRead < 10000)
                        uciParser.SetFilePosition(0); // go around again
                    records += recordsRead;
                    values.clear();
                    labels.clear();
                } while (records < 60000);
            })
            << " ms" << endl;
    }
    {
        vector<cntk::StreamDescriptor> streams(2);
        streams[0].m_alias = "F";
        streams[0].m_storageType = cntk::StorageType::dense;
        streams[0].m_sampleDimension = 784;

        streams[1].m_alias = "L";
        streams[1].m_storageType = cntk::StorageType::dense;
        streams[1].m_sampleDimension = 10;

        // Setting the chunk size to max to be apples-to-apples with UCI reader;
        cntk::CNTKTextFormatReaderTestRunner<float> textParser(textFilename, streams, SIZE_MAX, 1);

        const auto& timeline = textParser.GetTimeline();
        UNUSED(timeline);
        cout << "CNTKTextFormatReader : "
            << benchmark([&textParser]()
            {
                const auto& chunk = textParser.GetChunk(0);
                UNUSED(chunk);
            })
            << "ms" << endl;
    }
}

void compareParsingPerf(wstring uciFilename, wstring textFilename)
{
    cout << "Reading 150000 sequences. " << endl;
    {
        UCIParser<float, int> uciParser(char(0), char(0));
        size_t bufSize = 256 * 1024; // same buffer size as in CNTKTextFormatReader
        // uci parser only counts the number of lines, so using the same file for both
        uciParser.ParseInit(uciFilename.c_str(), 1, 784, 0, 1, bufSize);

        vector<float> values;
        values.reserve(784 * 10000);
        vector<int> labels;
        labels.reserve(10000);

        cout << "UCIFastReader : "
            << benchmark([&uciParser, &values, &labels]()
        {
            int records = 0;
            do
            {
                int recordsRead = uciParser.Parse(10000, &values, &labels);
                if (recordsRead < 10000)
                    uciParser.SetFilePosition(0); // go around again
                records += recordsRead;
                values.clear();
                labels.clear();
            } while (records < 150000);
        })
            << " ms" << endl;
    }
    {
        vector<cntk::StreamDescriptor> streams(2);
        streams[0].m_alias = "F";
        streams[0].m_storageType = cntk::StorageType::dense;
        streams[0].m_sampleDimension = 784;

        streams[1].m_alias = "L";
        streams[1].m_storageType = cntk::StorageType::dense;
        streams[1].m_sampleDimension = 10;

        // Using default chunking parameters;
        cntk::CNTKTextFormatReaderTestRunner<float> textParser(textFilename, streams, 32*1024 *1024, 3);

        const auto& timeline = textParser.GetTimeline();
        cout << "CNTKTextFormatReader : "
            << benchmark([&textParser, &timeline]()
        {
            size_t chunkId = 0;
            auto chunk = textParser.GetChunk(chunkId);
            size_t records = 0;
            do
            {
                const auto& descriptor = timeline[records % timeline.size()];
                if (chunkId != descriptor.m_chunkId)
                {
                    chunkId = descriptor.m_chunkId;
                    chunk = textParser.GetChunk(chunkId);
                }
                const auto& s =chunk->GetSequence(descriptor.m_id);
                records += (reinterpret_cast<cntk::DenseSequenceData*>(s[0].get()))->m_numberOfSamples;
            } while (records < 150000);
            
        })
            << "ms" << endl;
    }
}

int wmain(int argc, wchar_t *argv[])
{
    if (argc < 2)
    {
        RuntimeError("Please specify data directory");
    }

    //BOOST_ASSERT_MSG(false, "Please run the benchmark in the Release Mode");

    fs::path data_path(fs::initial_path<fs::path>());
    //current working directory
    data_path = fs::system_complete(fs::path(argv[1]));

    data_path /= "/";

    // In release mode the expected numbers are 500 ms for UCI and 150 for CNTKText
    compareIndexingPerf(data_path.generic_wstring() + L"cntk_text_format_data");

    // In release mode the expected numbers are 5000 ms for UCI and 1500 ms for CNTKText
    //compareIndexingPerf(data_path.generic_wstring() + L"cntk_text_format_data_x10");
    
    // In release mode the expected numbers are 1500 ms for UCI and 1700 ms for CNTKText
    compareParsingPerfSinglePass(data_path.generic_wstring() + L"uci_data",
        data_path.generic_wstring() + L"cntk_text_format_data");

    // In release mode the expected numbers are 3600 ms for UCI and 3700 
    // (1700 ms if the number of chunks to cache set to >= 4) ms for CNTKText
    compareParsingPerf(data_path.generic_wstring() + L"uci_data",
        data_path.generic_wstring() + L"cntk_text_format_data");

    cout << "Press any key to continue" << endl;
    std::cin.get();
    
    return 0;
}
