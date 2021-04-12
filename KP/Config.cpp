#include "Config.hpp"

std::istream& operator>>(std::istream &istream, Config& config)
{
    istream >> config.framesNum;

    istream >> config.outputTemplate;

    istream >> config.width >> config.height;

    istream >> config.horizontalViewDegrees;

    istream >> config.lookFrom >> config.lookAt;

    istream >> config.A >> config.B >> config.C;

    istream >> config.floorData;

    istream >> config.lightSourcesNum;

    for (int i = 0; i < config.lightSourcesNum; ++i)
        istream >> config.lightSources[i];

    istream >> config.recursionDepth;

    istream >> config.sqrtSamplesPerPixel;

    return istream;
}

std::istream& operator>>(std::istream &istream, Trajectory& trajectory)
{
    istream >> trajectory.r >> trajectory.z >> trajectory.phi;

    istream >> trajectory.rA >> trajectory.zA;

    istream >> trajectory.rOm >> trajectory.zOm >> trajectory.phiOm;

    istream >> trajectory.rP >> trajectory.zP;

    return istream;
}

std::istream& operator>>(std::istream &istream, FigureData& figureData)
{
    istream >> figureData.origin;

    istream >> figureData.color;

    istream >> figureData.radius;

    istream >> figureData.reflectance >> figureData.transparency;

    istream >> figureData.edgeLightsNum;

    return istream;
}

std::istream& operator>>(std::istream &istream, FloorData& floorData)
{
    istream >> floorData.A >> floorData.B >> floorData.C >> floorData.D;

    istream >> floorData.texturePath;

    istream >> floorData.color;

    istream >> floorData.reflectance;

    return istream;
}

std::istream& operator>>(std::istream &istream, LightSource& lightSource)
{
    istream >> lightSource.origin >> lightSource.radius >> lightSource.color;

    return istream;
}
