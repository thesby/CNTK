//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include "BpttPacker.h"
#include "ElementTypeUtils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

struct SequenceBuffer
{
    SequenceBuffer(size_t parallelNumberOfSequences)
    {
        m_preparedSequences.reserve(parallelNumberOfSequences);
        m_preparedSequenceLength.reserve(parallelNumberOfSequences);
        m_currentPositions.reserve(parallelNumberOfSequences);
    }

    bool NothingToPack() const
    {
        auto it = std::find_if(
            m_preparedSequences.begin(),
            m_preparedSequences.end(),
            [](const std::deque<SequenceDataPtr>& sequences) -> bool { return !sequences.empty(); });
        return it == m_preparedSequences.end();
    }

    // A matrix of prepared sequences. The number of rows(RN) = m_parallelNumberOfSequences
    // in each row we at least holding sequences to fill in the truncation length.
    // Only at the end of the epoch there could be less than truncation length number of samples in this matrix
    //
    // It looks something like that:
    //  /***s11***/ /***s12**/
    //  ....
    //  /**********sM1*********/
    //  ....
    // /*sRN1*//*sRN2*//*sRN2*/
    std::vector<std::deque<SequenceDataPtr>> m_preparedSequences;

    std::vector<size_t> m_preparedSequenceLength;
    std::vector<size_t> m_sequencePosition;
    std::vector<size_t> m_sequenceSamplePosition;

    // Describes the position in the sequences in the matrix above from where the minibatch begins.
    // m_currentPositions.size() == m_preparedSequences.size()
    std::vector<size_t> m_currentPositions;
};

BpttPacker::BpttPacker(
    MemoryProviderPtr memoryProvider,
    TransformerPtr transformer,
    size_t minibatchSize,
    size_t truncationSize,
    const std::vector<StreamDescriptionPtr>& streams) : m_transformer(transformer),
    m_minibatchSize(minibatchSize),
    m_outputStreams(streams),
    m_memoryProvider(memoryProvider),
    m_truncationSize(truncationSize)
{
    m_parallelNumberOfSequences = (size_t)std::floor(m_minibatchSize / truncationSize);
    m_inputStreams = m_transformer->GetStreamDescriptions();
    assert(m_inputStreams.size() == m_outputStreams.size());
    assert(
        std::find_if(
        m_outputStreams.begin(),
        m_outputStreams.end(),
        [](const StreamDescriptionPtr& s)
        {
            return s->m_storageType == StorageType::sparse_csc;
        }) == m_outputStreams.end());

    assert(m_minibatchSize > 0);
    for (int i = 0; i < m_outputStreams.size(); ++i)
    {
        const auto& stream = m_outputStreams[i];
        UNUSED(stream);

        // Input and output should match in everything except for sparse/dense.
        assert(stream->m_elementType == ElementType::tfloat || stream->m_elementType == ElementType::tdouble);
        assert(stream->m_name == m_inputStreams[i]->m_name);
        assert(stream->m_id == m_inputStreams[i]->m_id);
        assert(GetSampleSize(m_inputStreams[i]) == GetSampleSize(stream));

        m_streamBufferSizes.push_back(0);
        m_streamBuffers.push_back(nullptr);

        m_sequenceBufferPerStream.push_back(std::make_shared<SequenceBuffer>(m_parallelNumberOfSequences));
    }

    InitializePreparedSequences();
}

void BpttPacker::InitializePreparedSequences()
{
    //Initializes sequences.
}

Minibatch BpttPacker::ReadMinibatch()
{
    Minibatch result;

    // Currently all we expect sequences of identical length between different streams.
    if (m_sequenceBufferPerStream.front()->NothingToPack())
    {
        result.m_endOfEpoch = true;
        return result;
    }

    for (size_t streamIndex = 0; streamIndex < m_outputStreams.size(); ++streamIndex)
    {
        m_currentLayouts[streamIndex]->Init(m_parallelNumberOfSequences, m_truncationSize);
        for (size_t sequenceIndex = 0; sequenceIndex < m_parallelNumberOfSequences; ++sequenceIndex)
        {
            PackSequence(streamIndex, sequenceIndex);
        }

        StreamMinibatchPtr m = std::make_shared<StreamMinibatch>();
        m->m_data = m_streamBuffers[streamIndex].get();
        m->m_dataSize = m_currentLayouts[streamIndex]->GetActualNumSamples() * GetSampleSize(m_outputStreams[streamIndex]);
        m->m_layout = m_currentLayouts[streamIndex];
        result.m_data.push_back(m);
    }

    return result;
}

void BpttPacker::PackSequence(size_t streamIndex, size_t slotIndex)
{
    if (m_sequenceBufferPerStream[streamIndex]->m_preparedSequenceLength[slotIndex] < m_truncationSize)
    {
        GetSequencesToSlot(slotIndex);
    }

    size_t numberOfSamples = std::min(m_truncationSize, m_sequenceBufferPerStream[streamIndex]->m_preparedSequenceLength[slotIndex]);
    if (numberOfSamples == 0)
    {
        // Reached the end, simply return.
        return;
    }

    size_t sampleSize = 0;
    size_t stride = m_parallelNumberOfSequences * sampleSize;
    StorageType storageType = m_inputStreams[streamIndex]->m_storageType;
    size_t elementSize = GetSizeByType(m_inputStreams[streamIndex]->m_elementType);

    // Add initial sequence layouts???
    size_t& sequencePosition = m_sequenceBufferPerStream[streamIndex]->m_sequencePosition[slotIndex];
    size_t& samplePosition = m_sequenceBufferPerStream[streamIndex]->m_sequenceSamplePosition[slotIndex];

    m_currentLayouts[streamIndex]->AddSequence(
        NEW_SEQUENCE_ID,
        slotIndex,
        -(int)samplePosition,
        m_sequenceBufferPerStream[streamIndex]->m_preparedSequences[slotIndex][sequencePosition]->m_numberOfSamples);

    // Ok, now fill in the buffer.
    for (size_t i = 0; i < numberOfSamples; ++i)
    {
        if (samplePosition >= m_sequenceBufferPerStream[streamIndex]->m_preparedSequences[slotIndex][sequencePosition]->m_numberOfSamples)
        {
            // Starting a new sequence. Have to reset current pointers and add it to the minibatch.
            samplePosition = 0;
            m_sequenceBufferPerStream[streamIndex]->m_preparedSequences[slotIndex].pop_front();

            //Adding next sequence to the minibatch.
            m_currentLayouts[streamIndex]->AddSequence(
                NEW_SEQUENCE_ID,
                slotIndex,
                numberOfSamples,
                numberOfSamples + m_sequenceBufferPerStream[streamIndex]->m_preparedSequences[slotIndex][sequencePosition]->m_numberOfSamples);
        }

        auto data = m_sequenceBufferPerStream[streamIndex]->m_preparedSequences[slotIndex][sequencePosition];
        void* destination = m_streamBuffers[streamIndex].get() + stride * i + slotIndex * sampleSize;
        if (storageType == StorageType::dense)
        {
            PackDenseSample(destination, data, samplePosition, elementSize, sampleSize);
        }
        else // sparse
        {
            PackSparseSample(destination, data, samplePosition, elementSize, sampleSize);
        }
    }

    // Adding the last gap if there are 
    if (numberOfSamples < m_truncationSize)
    {
        m_currentLayouts[streamIndex]->AddSequence(
            GAP_SEQUENCE_ID,
            slotIndex,
            numberOfSamples,
            m_truncationSize - numberOfSamples);
    }
}


void BpttPacker::GetSequencesToSlot(size_t slotIndex)
{
    while (m_truncationSize > m_sequenceBufferPerStream.front()->m_preparedSequenceLength[slotIndex])
    {
        // We always need only a single sequence
        auto s = m_transformer->GetNextSequences(1);
        if (s.m_endOfEpoch)
        {
            break;
        }

        for (size_t i = 0; i < s.m_data.size(); ++i)
        {
            // expects only a single sequence
            assert(s.m_data[i].size() == 1);
            m_sequenceBufferPerStream[i]->m_preparedSequences[slotIndex].push_back(s.m_data[i].front());
            m_sequenceBufferPerStream[i]->m_preparedSequenceLength[slotIndex] += s.m_data[i].front()->m_numberOfSamples;
        }
    }
}

void BpttPacker::PackSparseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t /*sampleSize*/)
{
    SparseSequenceDataPtr s = static_pointer_cast<SparseSequenceData>(sequence);
    size_t nonZeroCount = s->m_indices[sample].size();
    for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
    {
        memcpy(
            (char*)destination + s->m_indices[sample][nonZeroIndex] * elementSize,
            (const char*)(s->m_data) + nonZeroIndex * elementSize,
            elementSize);
    }
}

void BpttPacker::PackDenseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t /*elementSize*/, size_t sampleSize)
{
    memcpy(destination, (char*)(sequence->m_data) + sample * sampleSize, sampleSize);
}

size_t BpttPacker::GetSampleSize(StreamDescriptionPtr stream)
{
    assert(stream != nullptr);
    size_t elementSize = GetSizeByType(stream->m_elementType);
    return stream->m_sampleLayout->GetNumElements() * elementSize;
}

std::shared_ptr<char> BpttPacker::AllocateBuffer(size_t numElements, size_t elementSize)
{
    return std::shared_ptr<char>(
        reinterpret_cast<char*>(m_memoryProvider->Alloc(elementSize, numElements)),
        [this](char* p)
    {
        m_memoryProvider->Free(p);
    });
}

}}}
