#include "denoiser.h"
#include<fstream>

Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Reproject
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.f);

            int id = frameInfo.m_id(x, y);
            if (id == -1) continue;

            Matrix4x4 world_to_local = Inverse(frameInfo.m_matrix[id]);
            Matrix4x4 pre_local_to_world = m_preFrameInfo.m_matrix[id];
            auto pre_world_position = pre_local_to_world(world_to_local(frameInfo.m_position(x, y), Float3::Point), Float3::Point);
            auto pre_screen_position = preWorldToScreen(pre_world_position, Float3::Point);
            if (pre_screen_position.x > width || pre_screen_position.x < 0 || pre_screen_position.y > height || pre_screen_position.y < 0) {
                continue;
            } else {
                if (id == m_preFrameInfo.m_id(pre_screen_position.x, pre_screen_position.y)) {
                    m_valid(x, y) = true;
                    m_misc(x, y) = m_accColor(pre_screen_position.x, pre_screen_position.y);
                }
            }
        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 3;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Temporal clamp
            Float3 color = m_accColor(x, y);
            // TODO: Exponential moving average
            float alpha = 1.0f;

            if (m_valid(x, y)) {
                alpha = m_alpha;

                int x_start = std::max(0, x - kernelRadius);
                int x_end = std::min(width - 1, x + kernelRadius);
                int y_start = std::max(0, y - kernelRadius);
                int y_end = std::min(height - 1, y + kernelRadius);

                Float3 mu(0.f);
                Float3 sigma(0.f);

                for (int m = x_start; m <= x_end; m++) {
                    for (int n = y_start; n <= y_end; n++) {
                        mu += curFilteredColor(m, n);
                        sigma += Sqr(curFilteredColor(x, y) - curFilteredColor(m, n));
                    }
                }

                int count = kernelRadius * 2 + 1;
                count *= count;

                mu /= float(count);
                sigma = SafeSqrt(sigma / float(count));
                color = Clamp(color, mu - sigma * m_colorBoxK, mu + sigma * m_colorBoxK);
            }

            m_misc(x, y) = Lerp(color, curFilteredColor(x, y), alpha);
        }
    }
    std::swap(m_misc, m_accColor);
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 16;

#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Joint bilateral filter
            int x_start = std::max(0, x - kernelRadius);
            int x_end = std::min(width - 1, x + kernelRadius);
            int y_start = std::max(0, y - kernelRadius);
            int y_end = std::min(height - 1, y + kernelRadius);

            Float3 normalj = frameInfo.m_normal(x, y);
            Float3 positionj = frameInfo.m_position(x, y);
            Float3 colorj = frameInfo.m_beauty(x, y);

            float total_weight = 0.f;
            Float3 final_color;

            for(int i = x_start; i <= x_end; i++) {
                for(int j = y_start; j <= y_end; j++) {
                    Float3 normali = frameInfo.m_normal(i, j);
                    Float3 positioni = frameInfo.m_position(i, j);
                    Float3 colori = frameInfo.m_beauty(i, j);

                    float Dposition = SqrDistance(positioni, positionj)  / 2.0f / m_sigmaCoord / m_sigmaCoord;
                    float Dcolor = SqrDistance(colorj , colori) / 2 / m_sigmaColor / m_sigmaColor;
                    float Dnormal = SafeAcos(Dot(normali, normalj));
                    Dnormal *= Dnormal;
                    Dnormal / (2.0f * m_sigmaNormal * m_sigmaNormal);
                    float Dplane = 0.f;
                    if (Dposition > 0.f)
                        Dplane = Dot(normalj, Normalize(positioni - positionj));
                    Dplane *= Dplane;
                    Dplane /= (2.0f * m_sigmaPlane * m_sigmaPlane);
                    float weight = std::exp(-Dplane - Dposition - Dcolor - Dnormal);
                    total_weight += weight;
                    final_color += colori * weight;
                }
            }

            filteredImage(x, y) = final_color / total_weight;
        }
    }
    return filteredImage;
} 

void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    filteredColor = Filter(frameInfo);

    // Reproject previous frame color to current
    if (m_useTemportal) {
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}
